import { BadRequestException, Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { startWorker } from 'src/worker/worker-manager';

@Injectable()
export class SystemConfigService {
  private concurrency: number;
  private startTime: Date;

  constructor(private readonly configService: ConfigService) {
    this.concurrency = Number(this.configService.get('BULL_CONCURRENCY') ?? 2);
    this.startTime = new Date();
  }

  getConcurrency() {
    return {concurrency: this.concurrency};  
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

    this.concurrency = value;

    await startWorker(value);

    return { message: 'Concurrency set successfully', concurrency: value };
  }
}
