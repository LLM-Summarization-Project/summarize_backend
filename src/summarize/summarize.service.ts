import { Injectable } from '@nestjs/common';
import { PrismaService } from './prisma/prisma.service';
import { randomUUID } from 'crypto';
import { QueueService } from 'src/queue/queue.service';
import path from 'path';
import * as fs from 'fs/promises';
import Redis from 'ioredis';
import { ConfigService } from '@nestjs/config';
import { VideoCacheService } from 'src/cache/video-cache.service';
import axios from 'axios';

@Injectable()
export class SummarizeService {
  private readonly redis: Redis;
  constructor(
    private prisma: PrismaService,
    private queueService: QueueService,
    private configService: ConfigService,
    private videoCacheService: VideoCacheService,
  ) {
    this.redis = new Redis({
      host: this.configService.get<string>('REDIS_HOST') ?? 'localhost',
      port: Number(this.configService.get<number>('REDIS_PORT') ?? 6379),
    });
  }

  async createSummary(youtubeUrl: string, userId: number, whisperTempParam?: number, youtubeApi?: boolean) {
    // Check if cache is disabled via env
    const disableCache = this.configService.get<string>('DISABLE_CACHE') === 'true';

    // Cache-first: Check if video already processed (space-based lookup)
    if (!disableCache) {
      const cached = await this.videoCacheService.getCachedSummary(youtubeUrl);
      if (cached) {
        console.log(`Cache HIT for ${youtubeUrl}, returning existing summary`);

        // Background: Try to assign user to topic if keyword exists
        this.prisma.summary.findUnique({ where: { id: cached.id }, select: { keyword: true } })
          .then(s => {
            if (s?.keyword) {
              this.postAssignUser(userId, s.keyword);
            }
          })
          .catch(err => console.error('Error fetching keyword for cache hit:', err));

        return { jobId: cached.id, status: 'CACHED' as const, fromCache: true };
      }
    } else {
      console.log(`Cache DISABLED, skipping cache lookup`);
    }

    // Check DB for existing completed summary (skip if cache disabled)
    if (!disableCache) {
      const existing = await this.prisma.summary.findFirst({
        where: { youtubeUrl, status: 'DONE' },
      });
      if (existing) {
        // Warm the cache for next time
        await this.videoCacheService.setCachedSummary(youtubeUrl, {
          id: existing.id,
          youtubeUrl: existing.youtubeUrl,
          status: 'DONE',
        });
        console.log(`DB HIT for ${youtubeUrl}, cached and returning`);

        if (existing.keyword) {
          this.postAssignUser(userId, existing.keyword);
        }

        return { jobId: existing.id, status: 'CACHED' as const, fromCache: true };
      }
    }

    // Cache miss - create new job
    const id = randomUUID();

    await this.prisma.summary.create({
      data: { id, youtubeUrl, status: 'QUEUED', owners: { create: { userId }} },
    });
    console.log(`Created summary record with ID: ${id}`);

    // ส่งงานเข้า BullMQ พร้อม whisperTemp
    // ถ้ามีค่าจาก request body ใช้ค่านั้น, ถ้าไม่มีอ่านจาก .env
    const whisperTemp = whisperTempParam ?? await this.readWhisperTempFromEnvFile();
    console.log(`Using WHISPER_TEMP: ${whisperTemp} (from ${whisperTempParam !== undefined ? 'request' : 'env'})`);
    if (!youtubeApi) {
      youtubeApi = false
    }
    await this.queueService.addRunJob({ summaryId: id, youtubeUrl, userId, whisperTemp, youtubeApi });

    return { jobId: id, status: 'QUEUED' as const };
  }

  // อ่าน WHISPER_TEMP จากไฟล์ .env โดยตรง (runtime, ไม่ cache)
  private async readWhisperTempFromEnvFile(): Promise<number> {
    try {
      const envPath = path.resolve(process.cwd(), '.env');
      const content = await fs.readFile(envPath, 'utf-8');
      const match = content.match(/^WHISPER_TEMP=(.+)$/m);
      if (match) {
        const val = parseFloat(match[1].trim());
        if (!isNaN(val)) return val;
      }
    } catch (e) {
      // fallback to process.env if file read fails
    }
    return parseFloat(process.env.WHISPER_TEMP ?? '0.0');
  }

  async getSummary(id: string) {
    const summary = await this.prisma.summary.findUnique({ where: { id } });

    if (!summary) {
      return { status: 'not_found' };
    }

    let summaryContent: string | null = null;
    if (summary.summaryPath && summary.status === 'DONE') {
      try {
        const normalizedPath = summary.summaryPath.replace(/\\/g, '/');
        const filepath = path.resolve(normalizedPath);
        summaryContent = await fs.readFile(filepath, 'utf-8');
      } catch (error) {
        console.error(`Failed to read summary file for ${id}:`, error);
      }
    }

    return { ...summary, summary: summaryContent };
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

  async getMySummary(userId: number) {
    const summaries = await this.prisma.summary.findMany({
      where: { owners: { some: { userId }}, status: 'DONE' },
      orderBy: { startedAt: 'desc' },
      select: {
        id: true,
        youtubeUrl: true,
        keyword: true,
        summaryPath: true,
        startedAt: true,
        status: true,
      },
    });

    const summariesWithContent = await Promise.all(
      summaries.map(async (summary) => {
        if (summary.summaryPath) {
          const normalizedPath = summary.summaryPath.replace(/\\/g, '/');
          const filepath = path.resolve(normalizedPath);

          try {
            const rawContent = await fs.readFile(filepath, 'utf-8');
            return { ...summary, summary: rawContent };
          } catch (error) {
            console.error(`Error reading file at ${filepath}:`, error);
            return { ...summary, summary: null };
          }
        }
        return { ...summary, summary: null };
      }),
    );
    return summariesWithContent;
  }

  async getAllSummary() {
    const summaries = await this.prisma.summary.findMany({
      orderBy: { startedAt: 'desc' },
      select: {
        id: true,
        youtubeUrl: true,
        percent: true,
        durationSec: true,
        startedAt: true,
        status: true,
        keyword: true,
        owners: { select: { 
          userId: true, 
          createdAt: true 
        }},
      },
    });

    const uniqueMap = new Map<string, typeof summaries[0]>();

    for (const summary of summaries) {
      const url = summary.youtubeUrl;
      
      const myEarliestTime = summary.owners.reduce((earliest, owner) => {
        return owner.createdAt < earliest ? owner.createdAt : earliest;
      }, summary.owners[0].createdAt);

      // loop เผื่อมี url ซ้ำกัน
      if (!uniqueMap.has(url)) {
        uniqueMap.set(url, summary)
      } else {
        const currentChamp = uniqueMap.get(url)!;
        
        const champEarliestTime = currentChamp.owners.reduce((earliest, owner) => {
          return owner.createdAt < earliest ? owner.createdAt : earliest;
        }, currentChamp.owners[0].createdAt);

        if (myEarliestTime < champEarliestTime) {
          uniqueMap.set(url, summary);
        }
      }
    }

    const uniqueSummaries = Array.from(uniqueMap.values()).map(summary => {
      return {
        ...summary,
        owner: summary.owners[0].userId, // Map oldest owner's ID to 'owner' field
      };
    });
    
    const activeWork = uniqueSummaries.filter((summary) => summary.status === 'RUNNING');

    return { summary: uniqueSummaries, active_worker: activeWork.length };
  }

  async cancelSummary(summaryId: string) {
    await this.prisma.summary.update({
      where: { id: summaryId },
      data: { status: 'CANCEL' },
    });
    return { success: true };
  }

  private async postAssignUser(userId: number, topicName: string) {
    console.log(`[postAssignUser] Attempting to assign user ${userId} to topic "${topicName}"...`);
    try {
      // Use process.env directly since Docker sets this env var
      const baseUrl = process.env.ONTOLOGY_SERVICE ?? 'http://localhost:3001';
      const response = await axios.post(`${baseUrl}/ontology/topic/assign-user`, {
        userId,
        topicName,
      });
      console.log(`[postAssignUser] SUCCESS: Assigned user ${userId} to topic "${topicName}". Status: ${response.status}`);
    } catch (error) {
      console.error(
        `[postAssignUser] FAILED: Could not assign user ${userId} to topic "${topicName}".`,
        `Error: ${error?.response?.status ?? 'Unknown Status'} - ${JSON.stringify(error?.response?.data ?? error?.message)}`
      );
    }
  }
}
