import { ApiProperty } from "@nestjs/swagger";

export class SummarizeRequestDto {
    @ApiProperty({
        example: 'https://youtu.be/Rq7plosixd0?si=-Xw05o5mTZd-eTBt',
        description: 'ลิงก์ Youtube ที่ต้องการสรุปเนื้อหา'
    })
    youtubeUrl: string;
}

export type SummaryStatus = 'QUEUED' | 'RUNNING' | 'DONE' | 'ERROR' | 'CANCEL';

export interface SummaryResult {
    transcript_path?: string;
    bullet_points_path?: string;
    articles_path?: string;
    scene_facts_path?: string;
}

export interface SummaryState {
    status: SummaryStatus;
    step?: string;
    percent?: number;
    result?: SummaryResult;
    errorMessage?: string;
    startedAt?: Date;
    finishedAt?: Date;
}