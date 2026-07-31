import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Endodontic Diagnostic & ECR CBCT Planning",
  description:
    "AAE clinical diagnosis and CBCT Patel ECR treatment planning. Educational use only.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
