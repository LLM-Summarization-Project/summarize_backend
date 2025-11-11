import { Inject, Injectable } from '@nestjs/common';
import { Queue } from 'bullmq';

export const SUMMARIZE_QUEUE = 'summarize';

@Injectable()
export class QueueService {
  public readonly queue: Queue;

  constructor() {
    this.queue = new Queue(SUMMARIZE_QUEUE, {
      connection: {
        host: process.env.REDIS_HOST ?? 'localhost',
        port: Number(process.env.REDIS_PORT ?? 6379),
      },
      defaultJobOptions: {
        attempts: 3, // retry 3 ครั้ง
        backoff: { type: 'exponential', delay: 10_000 }, // 10s, 20s, 40s...
        removeOnComplete: 1000,
        removeOnFail: 1000,
      },
    });
  }

  addRunJob(data: { summaryId: string; youtubeUrl: string }) {
    return this.queue.add('run', data, { jobId: data.summaryId });
  }
}
