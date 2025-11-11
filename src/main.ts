import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN ?? 'http://localhost:3000';

  app.enableCors({
    origin: FRONTEND_ORIGIN,                // ห้ามใช้ '*'
    credentials: true,                      // ต้องเปิดถ้าใช้ withCredentials
    methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    // ถ้า reverse proxy มีการ buffer SSE อาจต้อง expose header/ปิด buffer ด้วย
    // exposedHeaders: ['Content-Type'],
  });

  if (process.env.NODE_ENV !== 'production') {
    const config = new DocumentBuilder()
      .setTitle('LLM Summarizer API')
      .setDescription('API สำหรับสั่ง Summarize วิดีโอ YouTube ด้วย Python Pipeline')
      .setVersion('1.0')
      .addTag('summary')
      .build();
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api/docs', app, document);
  }

  await app.listen(process.env.PORT ?? 4001);
}
bootstrap();
