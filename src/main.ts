import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  const allowList = [
    process.env.FRONTEND_ORIGIN ?? 'http://localhost:8080',
    'http://127.0.0.1:8080',
    'http://25.28.124.88:8080',
    /^http:\/\/192\.168\.\d{1,3}\.\d{1,3}(:\d+)?$/,  // 192.168.x.x[:port]
    /^http:\/\/10\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$/, // 10.x.x.x
  ];

  app.enableCors({
    origin(origin, cb) {
      if (!origin) return cb(null, true); // curl/postman หรือ same-origin
      const ok = allowList.some((o) => (o instanceof RegExp ? o.test(origin) : o === origin));
      cb(ok ? null : new Error('CORS blocked'), ok);
    },
    credentials: true,                      // ต้องเปิดถ้าใช้ withCredentials
    methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    exposedHeaders: ['Content-Type'],
  });

  // app.enableCors({
  //   origin: "*",
  //   credentials: false,                      // ต้องเปิดถ้าใช้ withCredentials
  //   methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'],
  //   allowedHeaders: ['Content-Type', 'Authorization'],
  //   exposedHeaders: ['Content-Type'],
  // });

  if (process.env.NODE_ENV !== 'production') {
    const config = new DocumentBuilder()
      .setTitle('LLM Summarizer API')
      .setDescription('API สำหรับสั่ง Summarize วิดีโอ YouTube ด้วย Python Pipeline')
      .setVersion('1.0')
      .addTag('summary')
      .build();
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('swagger', app, document);
  }

  await app.listen(process.env.PORT ?? 8081);
}
bootstrap();
