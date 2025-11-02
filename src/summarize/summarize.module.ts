import { Module } from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeController } from './summarize.controller';

@Module({
  controllers: [SummarizeController],
  providers: [SummarizeService],
})
export class SummarizeModule {}
