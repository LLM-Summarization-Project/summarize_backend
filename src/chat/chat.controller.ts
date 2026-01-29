import { Controller, Get, Post, Body, Param, Query, Res, HttpCode, ParseIntPipe, UseGuards, Req } from '@nestjs/common';
import { ChatService } from './chat.service';
import { CreateChatDto } from './dto/create-chat.dto';
import type { Response } from 'express';
import { LocalJwtAuthGuard } from 'src/auth/jwt-auth.guard';
import { ApiBearerAuth, ApiCreatedResponse, ApiQuery } from '@nestjs/swagger';

@ApiBearerAuth()
@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @UseGuards(LocalJwtAuthGuard)
  @Post()
  @ApiCreatedResponse({
    description: 'Create a new chat',
    schema: {
      example: {
        "model": "llama3:8b",
        "created_at": "2026-01-27T19:18:10.9016104Z",
        "message": {
            "role": "assistant",
            "content": "สวัสดิการเมืองโลก: ความสำคัญของ Greenland ในยุคโดนัลด์ ทรัมพ์\n\nการที่นี่เมื่อวันที่ 14 พฤศจิกายน 2563 ทุกคนในโลกทุกประเทศเรียกว่าความสั่นสะท้อนจากนโยบายทางการเมืองและเศรษฐกิจของผู้นำประเทศหนึ่งได้ประกาศว่าจะคุมเงินรายได้จากการขายน้ำมันให้แก่สัปประยากรอีกประเทศหนึ่งเป็นครั้งแรกในประวัติศาสตร์โลก\n\nทรัมพ์ผู้ดำสรัฐนั้นสั่งเข้าไปควบคุมน้ำมันกว่า 50 ล้านบาเร็วในเวเนซีอลารวมไปถึงสร้างความสัสเตินพระทั่วยุโรปเมื่อประกาศว่า Greenland ต้องเป็นของอเมริกาสัฐ\n\nขณะเดียวกันนี้ นางมาเรียคอลิน่ามาชาโด้ ผู้นำฝ่ายค้านและเจ้าของรางวัลโนเบลสัติภาพปี 2025 ได้ออกมาเคลื่อนไหวในประเทศเวนซูลาแล้ว โดยเธอได้ให้สัมภาษณ์กับฟอคสนิวว่าเตรียมจะเดินทางกลับเวนซูลาโดยเร็วที่สุดพร้อมขอบคุณทรัมพ์ที่ค้นมาดูโรลลงได้\n\nขณะเดียวกันนี้ ทรंपเองก็ดูเหมือนว่าจะเปรกว่านิดหนึ่งเหมือนกัน โดยแบบว่าขอจัดการปัญหาภายในเวนซูลาให้เรียบร้อยก่อนที่จะมีกรอบเลือกตั้ง 30 วันที่หลายคนเรียกร้อง\n\nการเมืองในวินซูลาก็ยังเป็นประเด็นวิจารย์อย่างหนักเลยครับ โดยทางคาร์โลลายเลวิทมั่นใจว่าการได้มาซึ่ง Greenland คือพารกิจความมั่นคงระดับชาติเพื่อสกัดกัดอิทธิของประเทศรัสเสียและจีน\n\nขณะเดียวกันนี้ สตีเฟ้นมิลเลอร์ จาก CNN ได้บอกว่าไม่มีใครกล้าสู้สารัฐด้วยกำลังตะหารเพื่อยแย่งชิง Greenland อีกที่วิจารย์เด่นมากว่าดู Greenland เหมือนเป็นอนานิคมและไม่มีสักยภาพพอจะคุ้มครองพื้นที่ยุทธศาท\n\nตอนนี้ครับทีมที่ปรึกษาของในทรัมพ์กำลังหารือทางเลือกทุกรูปแบบและย้ำว่ามาตรการทางตหารยังคงเป็นทางเลือกเสมอ"
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 118791859600,
        "load_duration": 9727198400,
        "prompt_eval_count": 1875,
        "prompt_eval_duration": 3406274100,
        "eval_count": 608,
        "eval_duration": 105006768900
      }
    },
  })
  create(@Body() createChatDto: CreateChatDto, @Req() req) {
    const userId = req.user.id; 
    createChatDto.userId = userId;
    return this.chatService.create(createChatDto);
  }

  @UseGuards(LocalJwtAuthGuard)
  @Post('stream')
  @HttpCode(200)
  async stream(@Body() createChatDto: CreateChatDto, @Req() req, @Res() res: Response) {
    // Set SSE headers manually for POST request
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    const userId = req.user.id; 
    createChatDto.userId = userId;

    const observable = await this.chatService.createStream(createChatDto);
    
    observable.subscribe({
      next: (event) => {
        res.write(`data: ${JSON.stringify(event.data)}\n\n`);
      },
      error: (err) => {
        console.error('Stream error:', err);
        res.write(`data: ${JSON.stringify({ error: err.message, done: true })}\n\n`);
        res.end();
      },
      complete: () => {
        res.end();
      }
    });
  }

  @UseGuards(LocalJwtAuthGuard)
  @Get('/history/:summaryId')
  @ApiQuery({ name: 'topicId', required: false, type: String })
  @ApiCreatedResponse({
    description: 'Get all chat by summaryId, topicId',
    schema: {
      example: [
        {
          "id": 87,
          "sessionId": 15,
          "role": "USER",
          "content": "hello",
          "createdAt": "2026-01-27T16:00:57.877Z"
        },
        {
          "id": 88,
          "sessionId": 15,
          "role": "ASSISTANT",
          "content": "สวัสดีครับ! คุณต้องการให้ผมอธิบายข้อมูลเกี่ยวกับสภาพอากาศหรือไม่?",
          "createdAt": "2026-01-27T16:00:57.877Z"
        },
        {
          "id": 89,
          "sessionId": 15,
          "role": "USER",
          "content": "hello",
          "createdAt": "2026-01-27T16:06:32.659Z"
        },
        {
          "id": 90,
          "sessionId": 15,
          "role": "ASSISTANT",
          "content": "สวัสดีอีกครั้ง! ถ้าคุณมีคำถามเพิ่มเติมเกี่ยวกับสภาพอากาศหรือข้อมูลอื่น ๆ ที่คุณต้องการทราบ กรุณาแจ้งให้ผมทราบได้เลยครับ",
          "createdAt": "2026-01-27T16:06:32.659Z"
        }
      ],
    },
  })
  history(
    @Param('summaryId') summaryId: string,
    @Query('topicId') topicId: string | undefined,
    @Req() req
  ) {
    return this.chatService.history({summaryId, userId: req.user.id, topicId});
  }
}
