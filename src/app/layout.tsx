import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MNIST Topology Editor",
  description: "Visual topology editor and compiler workbench for MNIST neural networks",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
