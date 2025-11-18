import { Controller, Get, Post } from '@nestjs/common';
import { QueueService } from './queue.service';

@Controller('queue')
export class QueueController {
  constructor(private readonly queueService: QueueService) {}

  @Post('clear')
  async clearQueue() {
    await this.queueService.clearQueue();
    return { ok: true, message: 'queue cleared' };
  }

  @Get('status')
  async getQueueStatus() {
    return this.queueService.getQueueStatus();
  }
}