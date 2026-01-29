import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  Req,
  UseGuards,
} from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeRequestDto } from './dto/create-summarize.dto';
import { ApiBearerAuth, ApiCreatedResponse, ApiTags } from '@nestjs/swagger';
import { LocalJwtAuthGuard } from 'src/auth/jwt-auth.guard';

@ApiTags('summary')
@ApiBearerAuth()
@Controller('summary')
export class SummarizeController {
  constructor(private readonly summarizeService: SummarizeService) { }

  @UseGuards(LocalJwtAuthGuard)
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
  createSummary(@Body() summarizeRequestDto: SummarizeRequestDto, @Req() req) {
    console.log('[DEBUG] createSummary - req.user:', req.user);
    console.log('[DEBUG] createSummary - userId:', req.user.id);
    return this.summarizeService.createSummary(
      summarizeRequestDto.youtubeUrl,
      req.user.id,
      summarizeRequestDto.whisperTemp,
      summarizeRequestDto.youtubeApi
    );
  }

  @Get('all')
  @ApiCreatedResponse({
    description: 'Get all data of summary(duplicated for cache summary)',
    schema: {
      example: {
      "summary": [
          {
            "userId": 2,
            "createdAt": "2026-01-27T16:24:08.247Z",
            "summary": {
                "id": "afc2430a-528b-490e-97d1-cc53cdb070ae",
                "youtubeUrl": "https://youtube.com/shorts/lr_h4-U401I?si=qGOMVNMqVFzdHYqa",
                "percent": 100,
                "durationSec": 81.79,
                "startedAt": "2026-01-27T16:24:11.416Z",
                "status": "DONE",
                "keyword": "บังคับ"
            }
          } 
        ],
      "active_worker": 18
      }
    }
  })
  getAllSummary() {
    return this.summarizeService.getAllSummary();
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
        keyword: 'ภัยธรรมชาติ',
        summary:
          '**บทความภาษาไทยแบบเล่าเรื่อง**ประเทศฟิลิปปินซ์อยู่ในสถานการณ์อันตรายเนื่องจากภายุไต้ฝุ่นฟงว่องลูกที่ 21 ที่กำลังเคลื่อนไหวทางทิศตะวันออกเฉียงเหนือของประเทศทำให้มีพายุที่รุนแรงที่สุดที่ถล่มฟิลิปปินซ์ในปีนี้ได้รับการรายงานว่ามีผู้เสียชีวิตสองรายประชาชน 1 ล้านคนออกจากพื้นที่ก่อนที่พยุลูกนี้จะขึ้นฝั่งก็มีเหตุการต่างๆไม่หยุดยั้งที่สำคัญเพจัดการกรมอูตูนิยมวิทยาของไทยเราไม่ได้คาดการณ์ว่าจะมีอิทธิพลใดๆ ต่อไทยเพราะว่าทิตย์ทางของพายุหลังเขาลงทะเลเขาจะย้อนกลับไปทางใต้หวันแล้วก็จะเริ่มอ่อนกำลังลงแล้วก็จะกลายเป็นมวนอากาศเห็นครับ  (จากภาพ) ภายในภาพแสดงให้เห็นว่าน้ำมันทันลักเข้ามาเยอะมากแล้วเนี่ยครับคลื่นยักรมันซัดเข้ามาที่ฝั่งนะครับคุณกำลังคาบ้าดีนะใช่ครับผมอันนี้จุดนี้อยู่ที่คัตันดวนเนสก่อนดูซ่อนของฟิลิปปินซ์นะครับเป็นอีทิพลของซูเปอร์ไต้ฝุ่นฝ่งวองที่ขึ้นฝั่งทำให้พลื่นสูงซัดเข้ามาน้ำทันลักเข้ามาไปถนนเลยนะครับสำหรับประเทศไทยตอนบนมีฝนลดลง ยังคงมีฝนฟ้าขนองบังแห่งเนื่องจากความรุนแรงของอากาศตามปกคลุมด้านตะวันตกทางเหนือของประเทศและประเทศเมนม่านะคะขอให้ประชาชนบราเวณกล่าวระวังอันตรายจากฝนฟ้าขนองส่วนเกษตรกรควรป้องกันและระวังความเสียหายที่จะเกิดตอบผลผลิตทางการเกษตรไว้ด้วย  **ข้อคิด**  ภายุไต้ฝุ่นฟงว่องลูกที่ 21เป็นพยุลูกที่ใหญ่มากเลยใช่ ขอพ่ายแค่เดียวเขาจะเริ่มอ่อนกำลังลงเพราะว่าเข่าจะไปเจอมวนอากาศเห็นปัฐาของพายนะครับผมก็เนี่ยละค่ะอย่างที่เห็นว่าเป็นพยุลูกที่ใหญ่มากเลยนะ',
      },
    },
  })
  getOntologyData(@Param('id') id: string) {
    return this.summarizeService.getOntologyData(id);
  }

  // get my summary
  @UseGuards(LocalJwtAuthGuard)
  @Get()
  @ApiCreatedResponse({
    description: 'Get my summary',
    schema: {
      example: [
        {
            "id": "afc2430a-528b-490e-97d1-cc53cdb070ae",
            "youtubeUrl": "https://youtube.com/shorts/lr_h4-U401I?si=qGOMVNMqVFzdHYqa",
            "keyword": "บังคับ",
            "summaryPath": "/app/outputs/2/afc2430a-528b-490e-97d1-cc53cdb070ae/summary.txt",
            "startedAt": "2026-01-27T16:24:11.416Z",
            "status": "DONE",
            "summary": null
        },
        {
            "id": "796e092d-4496-4dee-a8f6-c80e1e8aa309",
            "youtubeUrl": "https://www.youtube.com/shorts/sLar1RY7yQs",
            "keyword": "ยูทูบเบอร์",
            "summaryPath": "/app/outputs/2/796e092d-4496-4dee-a8f6-c80e1e8aa309/summary.txt",
            "startedAt": "2026-01-27T16:07:01.502Z",
            "status": "DONE",
            "summary": null
        }
      ]
    },
  })
  getMySummary(@Req() req) {
    return this.summarizeService.getMySummary(req.user.id);
  }

  @Post(':id/cancel')
  @ApiCreatedResponse({
    description: 'Cancel summary job by summaryId',
    schema: {
      example: {
        status: 'CANCEL',
      },
    },
  })
  cancelSummary(@Param('id') id: string) {
    return this.summarizeService.cancelSummary(id);
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
