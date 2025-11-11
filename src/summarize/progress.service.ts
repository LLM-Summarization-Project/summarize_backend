import { Injectable } from '@nestjs/common';
import { Subject, Observable } from 'rxjs';

export interface ProgressEvent {
  jobId: string;
  percent: number;
  message?: string;
  done?: boolean;
  result?: any;
  error?: string;
}

@Injectable()
export class ProgressService {
  private channels = new Map<string, Subject<ProgressEvent>>();

  /** ✅ เปิด stream สำหรับ jobId ที่ frontend จะ subscribe */
  stream(jobId: string): Observable<ProgressEvent> {
    // console.log('🧷 Subscribed:', jobId);
    if (!this.channels.has(jobId)) {
      this.channels.set(jobId, new Subject());
    }
    return this.channels.get(jobId)!.asObservable();
  }

  /** ✅ ปล่อย event ใหม่ออกไป (เรียกจาก QueueEventsListener หรือ worker) */
  emit(jobId: string, payload: ProgressEvent) {
    // console.log('📤 Emit event to job', jobId, payload);
    if (!this.channels.has(jobId)) {
      this.channels.set(jobId, new Subject());
    }
    this.channels.get(jobId)!.next(payload);
  }

  /** ✅ ปิด stream เมื่องานเสร็จ */
  complete(jobId: string) {
    if (this.channels.has(jobId)) {
      this.channels.get(jobId)!.complete();
      this.channels.delete(jobId);
    }
  }
}