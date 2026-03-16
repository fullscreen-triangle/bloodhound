import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";

const NotFound = () => {
  return (
    <>
      <Head>
        <title>404 | Bloodhound</title>
      </Head>
      <main className="h-[75vh] w-full bg-dark">
        <Layout className="relative !bg-transparent !pt-16 flex flex-col items-center justify-center">
          <div className="text-[10rem] font-bold text-primary/20 leading-none md:text-[6rem]">404</div>
          <h1 className="text-3xl font-bold mb-4">Page Not Found</h1>
          <p className="text-muted mb-8">This trajectory does not converge to a valid state.</p>
          <Link href="/" className="btn-primary">Return Home</Link>
        </Layout>
      </main>
    </>
  );
};

export default NotFound;
