-- ลบ foreign key constraint ก่อน
ALTER TABLE "Summary" DROP CONSTRAINT IF EXISTS "Summary_userId_fkey";

-- ลบ column userId
ALTER TABLE "Summary" DROP COLUMN IF EXISTS "userId";