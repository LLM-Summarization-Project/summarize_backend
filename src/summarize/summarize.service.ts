import { Injectable } from '@nestjs/common';
import { SummaryState } from './dto/create-summarize.dto';
import { randomUUID } from 'crypto';
import * as path from 'path';
import { spawn } from 'child_process';
import { PrismaService } from './prisma/prisma.service';

@Injectable()
export class SummarizeService {
  constructor(private prisma: PrismaService) {}
  private summaries = new Map<string, SummaryState>();

  async createSummary(youtubeUrl: string) {
    const id = randomUUID();
    await this.prisma.summary.create({
      data: {
        id,
        youtubeUrl,
        status: 'QUEUED',
      },
    });
    // this.summaries.set(id, {
    //   status: 'QUEUED',
    // });
    setImmediate(() => this.summarizeVideo(id, youtubeUrl));
    return { jobId: id, status: 'QUEUED' as const };
  }

  getSummary(id: string) {
    return (
      this.prisma.summary.findUnique({
        where: { id },
      }) ?? 'not_found'
    );
  }

  private async summarizeVideo(summaryId: string, youtubeUrl: string) {
    await this.prisma.summary.update({
      where: { id: summaryId },
      data: { status: 'RUNNING', startedAt: new Date(), percent: 0 },
    });
    // this.summaries.set(summaryId, {
    //   status: 'RUNNING',
    // startedAt: new Date(),
    // percent: 0,
    // });

    const outputsDir = path.resolve(process.cwd(), 'outputs');
    const runnerPath = path.resolve(process.cwd(), 'python', 'runner.py');

    const py = spawn(
      process.env.PYTHON_BIN ?? 'python',
      [
        runnerPath,
        '--youtube_url',
        youtubeUrl,
        '--out_dir',
        outputsDir,
        '--scene_thresh',
        process.env.SCENE_THRESH ?? '0.6',
        '--language',
        process.env.LANGUAGE ?? 'th',
        '--asr_device',
        process.env.ASR_DEVICE ?? 'cpu',
        '--vl_device',
        process.env.VL_DEVICE ?? 'cpu',
        '--whisper_model',
        process.env.WHISPER_MODEL ?? 'large-v3-turbo',
        ...(process.env.OLLAMA_API
          ? ['--ollama_api', process.env.OLLAMA_API]
          : []),
        ...(process.env.OLLAMA_MODEL
          ? ['--ollama_model', process.env.OLLAMA_MODEL]
          : []),
        '--summary_id',
        summaryId,
      ],
      {
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
        cwd: process.cwd(),
        stdio: ['ignore', 'pipe', 'pipe', 'pipe'],
      },
    );

    if (!py.stdout || !py.stderr || !py.stdio[3]) {
      throw new Error('Python process missing stdio pipes');
    }

    py.stdout.setEncoding('utf-8');
    py.stderr.setEncoding('utf-8');

    let lastLine: string | null = null;
    let outBuf = '';
    let stderr = '';

    py.stdout.on('data', (chunk: string) => {
      outBuf += chunk;
      const lines = outBuf.split(/\r?\n/);
      outBuf = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        // เก็บบรรทัด JSON ล่าสุดไว้ (ไม่ parse ตอนนี้)
        lastLine = trimmed;
      }
    });

    // ---------- stderr: log ทั้งหมด ----------
    py.stderr.on('data', (d: string) => {
      const text = d.toString();
      console.error(text);
      stderr += text;
    });

    // ---------- fd3: progress realtime ----------
    const progress = py.stdio[3] as NodeJS.ReadableStream;
    progress.setEncoding('utf8');
    progress.on('data', async (chunk: string) => {
      for (const line of chunk.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const msg = JSON.parse(trimmed);
          if (msg?.type === 'progress') {
            const percent = Math.max(
              0,
              Math.min(100, Number(msg.percent) || 0),
            );
            const step =
              typeof msg.step === 'string' && msg.step.trim()
                ? msg.step
                : 'RUNNING';
            await this.prisma.summary.update({
              where: { id: summaryId },
              data: {
                status: 'RUNNING',
                percent,
              },
            });
            // this.summaries.set(summaryId, {
            //   ...prev,
            //   status: 'RUNNING',
            //   step,
            //   percent,
            // });
          }
        } catch {
          // non-JSON line -> ignore
        }
      }
    });

    // ---------- child process error ----------
    py.on('error', async (err) => {
      await this.prisma.summary.update({
        where: { id: summaryId },
        data: {
          status: 'ERROR',
          errorMessage: `spawn error: ${err?.message ?? String(err)}`,
          finishedAt: new Date(),
        },
      });
      // this.summaries.set(summaryId, {
      //   status: 'ERROR',
      //   errorMessage: `spawn error: ${err?.message ?? String(err)}`,
      //   finishedAt: new Date(),
      // });
    });

    // ---------- close: สรุปผล ----------
    py.on('close', async (code) => {
      // ถ้าค้างบรรทัดสุดท้ายไว้ใน buffer ให้ลองเก็บเป็น lastJsonLine
      if (outBuf.trim()) {
        lastLine = outBuf.trim();
      }

      const finishError = async (msg: string) => {
        await this.prisma.summary.update({
          where: {
            id: summaryId,
          },
          data: {
            status: 'ERROR',
            errorMessage: msg,
            finishedAt: new Date(),
          },
        });
        // this.summaries.set(summaryId, {
        //   status: 'ERROR',
        //   errorMessage: msg,
        //   finishedAt: new Date(),
        // });
      };

      if (code === 0) {
        if (!lastLine) {
          return finishError(
            'Python exited with code 0 but no final JSON was emitted.',
          );
        }
        try {
          const result = JSON.parse(lastLine);
          const metrics = result.metrics;
          // ถ้าฝั่ง Pythonส่ง status มา ให้เคารพ
          if (result?.status && result.status !== 'ok') {
            return finishError(
              result?.errorMessage ?? 'Python returned error status.',
            );
          }
          await this.prisma.summary.update({
            where: { id: summaryId },
            data: {
              status: 'DONE',
              percent: 100,
              finishedAt: new Date(),

              transcriptPath: result.transcript_path,
              bulletPath: result.bullets_path,
              summaryPath: result.article_path,
              sceneFactsPath: result.scene_facts_path,

              whisperModel: metrics.whisper_model,
              asrDevice: metrics.asr_device,
              vlDevice: metrics.vl_device,
              vlModel: metrics.vl_model,
              sceneThresh: metrics.scene_thresh,
              enableOcr: metrics.enable_ocr,

              frames: metrics.frames,
              bulletCount: metrics.bullets,
              transcriptWord: metrics.transcript_words,
              summaryWord: metrics.article_words,
              timeDownloadSec: metrics.t_download,
              timeSpeechtoTextSec: metrics.t_asr,
              timeCaptionImageSec: metrics.t_caption,
              timeSummarizeSec: metrics.t_summarize,
              timeTotal: metrics.t_total,
              durationSec: metrics.duration_sec,
            },
          });
          // this.summaries.set(summaryId, {
          // status: 'DONE',
          // result,
          // percent: 100,
          // step: 'finished',
          // finishedAt: new Date(),
          // });
        } catch (e: any) {
          return finishError(`Failed to parse final JSON: ${e?.message}`);
        }
      } else {
        // non-zero exit: ลอง parse lastJsonLine เผื่อเป็น {"status":"error", ...}
        if (lastLine) {
          try {
            const result = JSON.parse(lastLine);
            if (result?.status === 'error') {
              return finishError(result?.errorMessage ?? `exit ${code}`);
            }
          } catch {
            /* ignore */
          }
        }
        return finishError(stderr || `exit ${code}`);
      }
    });
  }
}
