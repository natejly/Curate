import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const formData = await req.formData();
  const files = formData.getAll("files");
  // TODO: Save files to disk or process as needed
  return NextResponse.json({ success: true, count: files.length });
}
