import 'dotenv/config';
import { Worker, Job } from 'bullmq';
import { PrismaClient } from '@prisma/client';
import { spawn } from 'child_process';
import * as path from 'path';

const prisma = new PrismaClient();
const QUEUE = 'summarize';

const concurrency = Number(process.env.BULL_CONCURRENCY ?? 2); // ปรับได้ตามทรัพยากร

const worker = new Worker(
  QUEUE,
  async (job: Job) => {
    const { summaryId, youtubeUrl } = job.data as {
      summaryId: string;
      youtubeUrl: string;
    };

    // mark RUNNING
    await prisma.summary.update({
      where: { id: summaryId },
      data: { status: 'RUNNING', startedAt: new Date(), percent: 0 },
    });

    const user = 'user1'; // temporary user
    const outputsDir = path.resolve(process.cwd(), 'outputs', user);
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
        stdio: ['ignore', 'pipe', 'pipe', 'pipe'], // fd3 = progress
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

    // 💡 helper: ดัก % จาก tqdm (เช่น " 27%|██▋       | 1552/5757 [...]")
    const handleTqdmChunk = async (chunk: string) => {
      const lines = chunk.split(/\r?\n/);
      for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;

        // จับตัวเลขก่อน % แล้วตามด้วย |
        const m = line.match(/(\d{1,3})%\s*\|/);
        if (!m) continue;

        const p = Number(m[1]);
        if (Number.isNaN(p)) continue;

        const subprogress = Math.max(0, Math.min(100, p));

        const percent = 10 + Math.floor((subprogress * 35) / 100); // tqdm ช่วง ASR = 10-45%

        await job.updateProgress({
          percent,
          step: 'ถอดเสียง',
          subprogress: subprogress,
        });

        await prisma.summary.update({
          where: { id: summaryId },
          data: { status: 'RUNNING', percent },
        });
      }
    };

    // STDOUT: เก็บบรรทัดสุดท้าย (เป็น JSON สรุป) + ดัก tqdm เผื่อมันพ่น stdout
    py.stdout.on('data', (chunk: string) => {
      outBuf += chunk;
      const lines = outBuf.split(/\r?\n/);
      outBuf = lines.pop() ?? '';
      for (const line of lines) {
        const t = line.trim();
        if (t) lastLine = t;
      }

      // ดัก % จาก tqdm ที่อาจโผล่ใน stdout
      void handleTqdmChunk(chunk);
    });

    // STDERR: เก็บ error log + ดัก tqdm (ส่วนใหญ่ tqdm อยู่ตรงนี้)
    py.stderr.on('data', (d: string) => {
      const text = d.toString();
      console.error(`[${summaryId}]`, text);
      stderr += text;

      // ดัก % จาก tqdm ที่พ่นบน stderr
      void handleTqdmChunk(text);
    });

    // FD3 = progress (JSON lines จาก pipeline)
    const progress = py.stdio[3] as NodeJS.ReadableStream;
    progress.setEncoding('utf8');
    progress.on('data', async (chunk: string) => {
      for (const line of chunk.split(/\r?\n/)) {
        const t = line.trim();
        if (!t) continue;
        try {
          const msg = JSON.parse(t);
          if (msg?.type === 'progress') {
            const percent = Math.max(
              0,
              Math.min(99, Number(msg.percent) || 0),
            );
            await job.updateProgress({
              percent,
              step: msg.step ?? '',
              subprogress: msg.subprogress ?? '',
            });
            await prisma.summary.update({
              where: { id: summaryId },
              data: { status: 'RUNNING', percent },
            });
          }
        } catch {
          // ignore non-JSON
        }
      }
    });

    // Promise จบเมื่อโปรเซสปิด
    await new Promise<void>((resolve, reject) => {
      py.on('error', async (err) => {
        await prisma.summary.update({
          where: { id: summaryId },
          data: {
            status: 'ERROR',
            errorMessage: `spawn error: ${err?.message ?? String(err)}`,
            finishedAt: new Date(),
          },
        });
        reject(err);
      });

      py.on('close', async (code) => {
        // flush บรรทัดสุดท้าย
        if (outBuf.trim()) lastLine = outBuf.trim();

        const finishError = async (msg: string) => {
          await prisma.summary.update({
            where: { id: summaryId },
            data: {
              status: 'ERROR',
              errorMessage: msg,
              finishedAt: new Date(),
            },
          });
        };

        if (code === 0) {
          if (!lastLine) {
            await finishError('Python exited 0 but no final JSON emitted.');
            return reject(new Error('no-final-json'));
          }
          try {
            const result = JSON.parse(lastLine);
            const metrics = result.metrics;
            if (result?.status && result.status !== 'ok') {
              await finishError(
                result?.errorMessage ?? 'Python returned error status.',
              );
              return reject(new Error('python-error-status'));
            }

            await prisma.summary.update({
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
                keyword: metrics.keyword,
                timeDownloadSec: metrics.t_download,
                timeSpeechtoTextSec: metrics.t_asr,
                timeCaptionImageSec: metrics.t_caption,
                timeSummarizeSec: metrics.t_summarize,
                timeTotal: metrics.t_total,
                durationSec: metrics.duration_sec,
              },
            });
            await job.updateProgress({
              percent: 100,
              step: 'บันทึกข้อมูล',
              subprogress: 100,
            });

            resolve();
          } catch (e: any) {
            await finishError(`Failed to parse final JSON: ${e?.message}`);
            reject(e);
          }
        } else {
          if (lastLine) {
            try {
              const r = JSON.parse(lastLine);
              if (r?.status === 'error') {
                await finishError(r?.errorMessage ?? `exit ${code}`);
                return reject(new Error(r?.errorMessage ?? `exit ${code}`));
              }
            } catch {
              /* ignore */
            }
          }
          await finishError(stderr || `exit ${code}`);
          reject(new Error(stderr || `exit ${code}`));
        }
      });
    });

    // สำเร็จ
    return true;
  },
  {
    concurrency, // <<<< รันพร้อมกันได้เท่านี้
    connection: {
      host: process.env.REDIS_HOST ?? 'localhost',
      port: Number(process.env.REDIS_PORT ?? 6379),
    },
  },
);

// optional: log event
worker.on('completed', (job) => {
  console.log(`[OK] job ${job.id}`);
});
worker.on('failed', (job, err) => {
  console.error(`[FAIL] job ${job?.id}:`, err?.message);
});

// graceful shutdown
process.on('SIGINT', async () => {
  console.log('Shutting down worker...');
  await worker.close();
  await prisma.$disconnect();
  process.exit(0);
});
