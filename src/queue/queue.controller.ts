import { Controller, Get, Post } from '@nestjs/common';
import { QueueService } from './queue.service';
import { ApiCreatedResponse } from '@nestjs/swagger';

@Controller('queue')
export class QueueController {
  constructor(private readonly queueService: QueueService) {}

  @Post('clear')
  @ApiCreatedResponse({
    description: 'Clear queue',
    schema: {
      example: {
        ok: true,
        message: 'queue cleared',
      },
    },
  })
  async clearQueue() {
    await this.queueService.clearQueue();
    return { ok: true, message: 'queue cleared' };
  }

  @Get('status')
  @ApiCreatedResponse({
    description: 'Get queue status',
    schema: {
      example: {
        "waiting": 0,
        "active": 0,
        "completed": 0,
        "failed": 1,
        "delayed": 0,
        "paused": 0,
        "concurrency": 2,
        "freeSlots": 2,
        "isBusy": false
      },
    },
  })
  async getQueueStatus() {
    return this.queueService.getQueueStatus();
  }
}