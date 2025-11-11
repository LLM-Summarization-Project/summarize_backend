import { Injectable } from '@nestjs/common';
import { PrismaService } from './prisma/prisma.service';
import { randomUUID } from 'crypto';
import { QueueService } from 'src/queue/queue.service';
import path from 'path';
import * as fs from 'fs/promises';

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

  async getSummary(id: string) {
    const summary = await this.prisma.summary.findUnique({ where: { id } });

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
}
