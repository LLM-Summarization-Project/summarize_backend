import { Controller, Get, Post, Body, Patch, Param, Delete } from '@nestjs/common';
import { SystemConfigService } from './system-config.service';
import { SetConcurrencyDto } from './dto/set-concurrency.dto';
import { ApiCreatedResponse } from '@nestjs/swagger';

@Controller('system-config')
export class SystemConfigController {
  constructor(private readonly systemConfigService: SystemConfigService) {}

  @Get('concurrency')
  @ApiCreatedResponse({
    description: 'Get concurrency',
    schema: {
      example: {
        concurrency: 2,
      },
    },
  })
  getConcurrency() {
    return this.systemConfigService.getConcurrency();
  }

  @Get('uptime')
  @ApiCreatedResponse({
    description: 'Get uptime',
    schema: {
      example: {
        uptime: 15,
        startTime: "2026-01-27T19:11:54.739Z"
      },
    },
  })
  getUptime() {
    return this.systemConfigService.getUptime();
  }

  @Post('concurrency')
  @ApiCreatedResponse({
    description: 'Set concurrency',
    schema: {
      example: {
        message: "Concurrency set successfully",
        concurrency: 2
      },
    },
  })
  async setConcurrency(@Body() dot: SetConcurrencyDto) {
    return await this.systemConfigService.setConcurrency(dot.value);
  }
}
