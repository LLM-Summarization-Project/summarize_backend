import { Module } from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeController } from './summarize.controller';
import { PrismaModule } from './prisma/prisma.module';
import { QueueModule } from 'src/queue/queue.module';
import { AuthModule } from 'src/auth/auth.module';

@Module({
  imports: [AuthModule, PrismaModule, QueueModule],
  controllers: [SummarizeController],
  providers: [SummarizeService],
})
export class SummarizeModule {}
