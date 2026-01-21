import { Controller, Get, Post, Body, Param, Query, Res, HttpCode, ParseIntPipe, UseGuards, Req } from '@nestjs/common';
import { ChatService } from './chat.service';
import { CreateChatDto } from './dto/create-chat.dto';
import type { Response } from 'express';
import { LocalJwtAuthGuard } from 'src/auth/jwt-auth.guard';

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @UseGuards(LocalJwtAuthGuard)
  @Post()
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
  history(
    @Param('summaryId') summaryId: string,
    @Req() req
  ) {
    return this.chatService.history({summaryId, userId: req.user.id});
  }
}
