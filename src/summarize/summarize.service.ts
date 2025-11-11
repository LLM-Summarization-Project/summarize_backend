import { Injectable } from '@nestjs/common';
import { PrismaService } from './prisma/prisma.service';
import { randomUUID } from 'crypto';
import { QueueService } from 'src/queue/queue.service';

@Injectable()
export class SummarizeService {
  constructor(
    private prisma: PrismaService,
    private queueService: QueueService,
  ) {}

  async createSummary(youtubeUrl: string) {
    const id = randomUUID();

    await this.prisma.summary.create({
      data: { id, youtubeUrl, status: 'QUEUED' },
    });

    // ส่งงานเข้า BullMQ
    await this.queueService.addRunJob({ summaryId: id, youtubeUrl });

    return { jobId: id, status: 'QUEUED' as const };
  }

  getSummary(id: string) {
    return this.prisma.summary.findUnique({ where: { id } }) ?? 'not_found';
  }

  getOntologyData(id: string) {
    return (
      this.prisma.summary.findUnique({
        where: { id },
        select: { keyword: true, summaryPath: true },
      }) ?? 'not_found'
    );
  }
}
