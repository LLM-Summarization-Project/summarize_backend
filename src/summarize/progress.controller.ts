// src/summarize/progress.controller.ts
import { Controller, MessageEvent, Param, Sse } from '@nestjs/common';
import { map } from 'rxjs/operators';
import { Observable } from 'rxjs';
import { ProgressService, ProgressEvent } from './progress.service';

@Controller('jobs')
export class ProgressController {
  constructor(private readonly progress: ProgressService) {}

  @Sse(':id/stream')
  streamJob(@Param('id') id: string): Observable<MessageEvent> {
    return this.progress.stream(id).pipe(map((data: ProgressEvent): MessageEvent => ({ data })));
  }
}
