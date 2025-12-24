import { Controller, Get, Post, Body, Patch, Param, Delete } from '@nestjs/common';
import { SystemConfigService } from './system-config.service';
import { SetConcurrencyDto } from './dto/set-concurrency.dto';

@Controller('system-config')
export class SystemConfigController {
  constructor(private readonly systemConfigService: SystemConfigService) {}

  @Get('concurrency')
  getConcurrency() {
    return this.systemConfigService.getConcurrency();
  }

  @Get('uptime')
  getUptime() {
    return this.systemConfigService.getUptime();
  }

  @Post('concurrency')
  async setConcurrency(@Body() dot: SetConcurrencyDto) {
    return await this.systemConfigService.setConcurrency(dot.value);
  }
}
