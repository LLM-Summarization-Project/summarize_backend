import { Module } from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeController } from './summarize.controller';
import { PrismaModule } from './prisma/prisma.module';
import { QueueModule } from 'src/queue/queue.module';

@Module({
  imports: [PrismaModule, QueueModule],
  controllers: [SummarizeController],
  providers: [SummarizeService],
})
export class SummarizeModule {}
