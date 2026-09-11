import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Endodontic Diagnostic Agent",
  description:
    "AAE pulpal and apical diagnosis from structured clinical findings. Educational use only.",
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
