// src/summarize/progress.controller.ts
import { Controller, MessageEvent, Param, Sse } from '@nestjs/common';
import { map } from 'rxjs/operators';
import { Observable } from 'rxjs';
import { ProgressService, ProgressEvent } from './progress.service';
import { ApiCreatedResponse } from '@nestjs/swagger';

@Controller('jobs')
export class ProgressController {
  constructor(private readonly progress: ProgressService) {}

  @Sse(':id/stream')
  @ApiCreatedResponse({
    description: 'Stream job progress',
    schema: {
      example: {
        data: {
          jobId: '24aee79e-3560-488d-b05b-da801ab44c0b',
          status: 'PROCESSING',
          percent: 50,
          message: 'ถอดเสียง',
        },
      },
    },
  })
  streamJob(@Param('id') id: string): Observable<MessageEvent> {
    return this.progress.stream(id).pipe(map((data: ProgressEvent): MessageEvent => ({ data })));
  }
}
