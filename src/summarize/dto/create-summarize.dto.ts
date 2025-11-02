import { ApiProperty } from "@nestjs/swagger";

export class SummarizeRequestDto {
    @ApiProperty({
        example: 'https://youtu.be/Rq7plosixd0?si=-Xw05o5mTZd-eTBt',
        description: 'ลิงก์ Youtube ที่ต้องการสรุปเนื้อหา'
    })
    youtubeUrl: string;
}

export type SummaryStatus = 'queued' | 'running' | 'done' | 'error' | 'not_found';

export interface SummaryResult {
    transcript_path?: string;
    bullet_points_path?: string;
    articles_path?: string;
    scene_facts_path?: string;
}

export interface SummaryState {
    status: SummaryStatus;
    result?: SummaryResult;
    errorMessage?: string;
    startedAt?: Date;
    finishedAt?: Date;
}