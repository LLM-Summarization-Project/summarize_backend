import { Controller, Get, Post, Body, Param, Res, HttpCode } from '@nestjs/common';
import { ChatService } from './chat.service';
import { CreateChatDto } from './dto/create-chat.dto';
import type { Response } from 'express';

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post()
  create(@Body() createChatDto: CreateChatDto) {
    return this.chatService.create(createChatDto);
  }

  @Post('stream')
  @HttpCode(200)
  async stream(@Body() createChatDto: CreateChatDto, @Res() res: Response) {
    // Set SSE headers manually for POST request
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

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

  @Get('/history/:summaryId')
  history(@Param('summaryId') summaryId: string) {
    return this.chatService.history(summaryId);
  }
}
