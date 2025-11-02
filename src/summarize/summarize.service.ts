import { Injectable } from '@nestjs/common';
import { SummaryState, SummarizeRequestDto } from './dto/create-summarize.dto';
import { UpdateSummarizeDto } from './dto/update-summarize.dto';
import { randomUUID } from 'crypto';
import * as path from 'path';
import { spawn } from 'child_process';

@Injectable()
export class SummarizeService {
  private summaries = new Map<string, SummaryState>();

  async createSummary(youtubeUrl: string) {
    const id = randomUUID();
    this.summaries.set(id, {
      status: 'queued',
    });
    setImmediate(() => this.summarizeVideo(id, youtubeUrl));
    return { jobId: id, status: 'queued' as const };
  }

  getSummary(id: string) {
    return this.summaries.get(id) ?? { status: 'not_found' as const };
  }

  private summarizeVideo(summaryId: string, youtubeUrl: string) {
    this.summaries.set(summaryId, { status: 'running', startedAt: new Date() });

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
      },
    );

    let stdout = '',
      stderr = '';
    py.stdout.on('data', (d) => (stdout += d.toString()));
    py.stderr.on('data', (d) => {
      console.error(d.toString());
      stderr += d.toString();
    });

    py.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout || '{}');
          this.summaries.set(summaryId, {
            status: 'done',
            result: result,
            finishedAt: new Date(),
          });
        } catch (e: any) {
          this.summaries.set(summaryId, {
            status: 'error',
            errorMessage: `Failed to parse summary result: ${e?.message}`,
            finishedAt: new Date(),
          });
        }
      } else {
        this.summaries.set(summaryId, {
          status: 'error',
          errorMessage: stderr || `exit ${code}`,
          finishedAt: new Date(),
        });
      }
    });
  }

  // findAll() {
  //   return `This action returns all summarize`;
  // }

  // findOne(id: number) {
  //   return `This action returns a #${id} summarize`;
  // }

  // update(id: number, updateSummarizeDto: UpdateSummarizeDto) {
  //   return `This action updates a #${id} summarize`;
  // }

  // remove(id: number) {
  //   return `This action removes a #${id} summarize`;
  // }
}
