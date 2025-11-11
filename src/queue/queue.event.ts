// src/queue/queue.events.ts
import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { QueueEvents } from 'bullmq';
import { ProgressService } from 'src/summarize/progress.service';
import { SUMMARIZE_QUEUE } from './queue.service';

@Injectable()
export class QueueEventsListener implements OnModuleInit, OnModuleDestroy {
  private queueEvents: QueueEvents;

  constructor(private readonly progressService: ProgressService) {}

  async onModuleInit() {
    this.queueEvents = new QueueEvents(SUMMARIZE_QUEUE, {
      connection: {
        host: process.env.REDIS_HOST ?? 'localhost',
        port: Number(process.env.REDIS_PORT ?? 6379),
      },
    });

    // ฟัง event progress
    this.queueEvents.on('progress', ({ jobId, data }: { jobId: string, data: any}) => {
      console.log('Progress event:', jobId, data);
      this.progressService.emit(String(jobId), { jobId: String(jobId), ...data });
    });

    // ฟัง event completed
    this.queueEvents.on('completed', ({ jobId, returnvalue }) => {
      this.progressService.emit(String(jobId), {
        jobId: String(jobId),
        percent: 100,
        message: 'done',
        result: returnvalue,
      });
      this.progressService.complete(String(jobId));
    });

    // ฟัง event failed
    this.queueEvents.on('failed', ({ jobId, failedReason }) => {
      this.progressService.emit(String(jobId), {
        jobId: String(jobId),
        percent: 100,
        message: 'failed',
        error: failedReason,
      });
      this.progressService.complete(String(jobId));
    });

    console.log('✅ QueueEvents listener started.');
  }

  async onModuleDestroy() {
    await this.queueEvents.close();
  }
}
