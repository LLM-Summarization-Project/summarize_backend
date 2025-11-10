import { Module } from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeController } from './summarize.controller';
import { PrismaModule } from './prisma/prisma.module';

@Module({
  imports: [PrismaModule],
  controllers: [SummarizeController],
  providers: [SummarizeService],
})
export class SummarizeModule {}
