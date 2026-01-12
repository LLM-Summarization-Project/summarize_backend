import { BadRequestException, Injectable, OnModuleInit } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';
import { getConcurrency, setConcurrency, getRedisConfig } from '../shared/redis-concurrency';

@Injectable()
export class SystemConfigService implements OnModuleInit {
  private redis: Redis;
  private startTime: Date;

  constructor(private readonly configService: ConfigService) {
    this.redis = new Redis(getRedisConfig());
    this.startTime = new Date();
  }

  async onModuleInit() {
    // Initialize concurrency in Redis if not set
    const current = await getConcurrency(this.redis);
    const envConcurrency = Number(this.configService.get('BULL_CONCURRENCY') ?? 2);
    
    // Only set if Redis doesn't have a value yet (first startup)
    const exists = await this.redis.exists('system:concurrency');
    if (!exists) {
      await setConcurrency(this.redis, envConcurrency);
      console.log(`[SystemConfig] Initialized concurrency to ${envConcurrency}`);
    } else {
      console.log(`[SystemConfig] Current concurrency: ${current}`);
    }
  }

  async getConcurrency() {
    const concurrency = await getConcurrency(this.redis);
    return { concurrency };
  }

  getUptime() {
    const now = new Date();
    const uptimeMs = now.getTime() - this.startTime.getTime();
    const uptimeSeconds = Math.floor(uptimeMs / 1000);
    return { uptime: uptimeSeconds, startTime: this.startTime };
  }

  async setConcurrency(value: number) {
    if (value < 1 || value > 5) {
      throw new BadRequestException('Invalid concurrency value');
    }

    await setConcurrency(this.redis, value);

    return { message: 'Concurrency set successfully', concurrency: value };
  }
}
