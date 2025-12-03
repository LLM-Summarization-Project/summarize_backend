import { Injectable } from '@nestjs/common';
import { PrismaService } from './prisma/prisma.service';
import { randomUUID } from 'crypto';
import { QueueService } from 'src/queue/queue.service';
import path from 'path';
import * as fs from 'fs/promises';
import Redis from 'ioredis';

@Injectable()
export class SummarizeService {
  private readonly redis: Redis;
  constructor(
    private prisma: PrismaService,
    private queueService: QueueService,
  ) {
    this.redis = new Redis({
      host: process.env.REDIS_HOST ?? 'localhost',
      port: Number(process.env.REDIS_PORT ?? 6379),
    });
  }

  async createSummary(youtubeUrl: string) {
    const id = randomUUID();

    await this.prisma.summary.create({
      data: { id, youtubeUrl, status: 'QUEUED' },
    });
    console.log(`Created summary record with ID: ${id}`);

    // ส่งงานเข้า BullMQ
    await this.queueService.addRunJob({ summaryId: id, youtubeUrl });

    return { jobId: id, status: 'QUEUED' as const };
  }

  async getSummary(id: string) {
    const summary = await this.prisma.summary.findUnique({ where: { id } });
    // console.log(summary)

    if (!summary) {
      return { status: 'not_found' };
    }
    return summary;
  }

  async getOntologyData(id: string) {
    const data = await this.prisma.summary.findUnique({
      where: { id },
      select: { keyword: true, summaryPath: true },
    });

    if (!data) {
      return { status: 'not_found' };
    }

    const normalizedPath = data.summaryPath?.replace(/\\/g, '/');

    const filepath = path.resolve(normalizedPath || '');
    const rawContent = await fs.readFile(filepath, 'utf-8').catch(() => null);
    // console.log('filepath:', filepath);
    // console.log('rawContent:', rawContent);

    if (!filepath) {
      return {
        status: 'summary_file_not_found',
      };
    }

    const content = rawContent?.replace(/\r\n/g, '');
    // console.log('content:', content);

    return {
      keyword: data.keyword,
      summary: content,
    };
  }

  async cancelSummary(summaryId: string) {
    await this.prisma.summary.update({
      where: { id: summaryId },
      data: { status: 'CANCEL' },
    });
    return { success: true };
  }
}
