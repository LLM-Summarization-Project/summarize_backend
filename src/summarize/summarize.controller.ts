import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
} from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeRequestDto } from './dto/create-summarize.dto';
import { UpdateSummarizeDto } from './dto/update-summarize.dto';
import { ApiCreatedResponse, ApiTags } from '@nestjs/swagger';

@ApiTags('summary')
@Controller('summary')
export class SummarizeController {
  constructor(private readonly summarizeService: SummarizeService) {}

  @Post()
  @ApiCreatedResponse({
    description: 'Summary job created successfully',
    schema: {
      example: {
        jobId: '24aee79e-3560-488d-b05b-da801ab44c0b',
        status: 'QUEUED',
      },
    },
  })
  createSummary(@Body() summarizeRequestDto: SummarizeRequestDto) {
    return this.summarizeService.createSummary(summarizeRequestDto.youtubeUrl);
  }

  @Get(':id')
  @ApiCreatedResponse({
    description: 'Get all data of summary by summaryId',
    schema: {
      example: {
        id: '24aee79e-3560-488d-b05b-da801ab44c0b',
        youtubeUrl: 'https://youtu.be/lqCgb7pgbRQ?si=2NnYqKz7RAiKkugS',
        status: 'DONE',
        percent: 100,
        startedAt: '2025-11-11T10:25:41.921Z',
        finishedAt: '2025-11-11T10:33:39.824Z',
        transcriptPath:
          'C:\\Users\\North\\Desktop\\college\\year4sem1\\project prep\\summarize-backend\\outputs\\user1\\24aee79e-3560-488d-b05b-da801ab44c0b\\transcription.txt',
        bulletPath:
          'C:\\Users\\North\\Desktop\\college\\year4sem1\\project prep\\summarize-backend\\outputs\\user1\\24aee79e-3560-488d-b05b-da801ab44c0b\\dropdown_list.txt',
        summaryPath:
          'C:\\Users\\North\\Desktop\\college\\year4sem1\\project prep\\summarize-backend\\outputs\\user1\\24aee79e-3560-488d-b05b-da801ab44c0b\\summary.txt',
        sceneFactsPath:
          'C:\\Users\\North\\Desktop\\college\\year4sem1\\project prep\\summarize-backend\\outputs\\user1\\24aee79e-3560-488d-b05b-da801ab44c0b\\scene_facts.json',
        keyword: 'สแกมเมอร์',
        whisperModel: 'large-v3-turbo',
        asrDevice: 'cpu',
        vlDevice: 'cpu',
        vlModel: 'microsoft/Florence-2-base',
        sceneThresh: 0.6,
        enableOcr: false,
        frames: 6,
        bulletCount: 1,
        transcriptWord: 1074,
        summaryWord: 265,
        timeDownloadSec: 9.8,
        timeSpeechtoTextSec: 300.27,
        timeCaptionImageSec: 75.71,
        timeSummarizeSec: 80.79,
        timeTotal: 466.63,
        durationSec: 317.56,
        errorMessage: null,
      },
    },
  })
  getSummary(@Param('id') id: string) {
    return this.summarizeService.getSummary(id);
  }

  @Get(':id/ontology')
  @ApiCreatedResponse({
    description: 'Get(by summaryId) keyword and summary filepath',
    schema: {
      example: {
        keyword: 'สแกมเมอร์',
        summaryPath:
          'C:/Users/North/Desktop/college/year4sem1/project prep/summarize-backend/outputs/user1/24aee79e-3560-488d-b05b-da801ab44c0b/summary.txt',
      },
    },
  })
  getOntologyData(@Param('id') id: string) {
    return this.summarizeService.getOntologyData(id);
  }

  // @Get()
  // findAll() {
  //   return this.summarizeService.findAll();
  // }

  // @Get(':id')
  // findOne(@Param('id') id: string) {
  //   return this.summarizeService.findOne(+id);
  // }

  // @Patch(':id')
  // update(@Param('id') id: string, @Body() updateSummarizeDto: UpdateSummarizeDto) {
  //   return this.summarizeService.update(+id, updateSummarizeDto);
  // }

  // @Delete(':id')
  // remove(@Param('id') id: string) {
  //   return this.summarizeService.remove(+id);
  // }
}
