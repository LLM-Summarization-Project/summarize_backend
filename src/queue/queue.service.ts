import { Inject, Injectable } from '@nestjs/common';
import { Queue } from 'bullmq';

export const SUMMARIZE_QUEUE = 'summarize';

@Injectable()
export class QueueService {
  public readonly queue: Queue;
  private readonly concurrency: number;

  constructor() {
    this.concurrency = Number(process.env.BULL_CONCURRENCY ?? 2);

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

  async clearQueue() {
    await this.queue.drain(true);
    await this.queue.clean(0, 0); // completed
    await this.queue.clean(0, 1); // failed
  }

  async getQueueStatus() {
    const queues = await this.queue.getJobCounts(
      'waiting',
      'active',
      'completed',
      'failed',
      'delayed',
      'paused',
    );
    const active = queues.active || 0;
    const waiting = queues.waiting || 0;

    const freeSlots = Math.max(this.concurrency - active, 0);

    return {
      ...queues,
      concurrency: this.concurrency,
      active,
      waiting,
      freeSlots,
      isBusy: freeSlots === 0,
    }
  }
}
