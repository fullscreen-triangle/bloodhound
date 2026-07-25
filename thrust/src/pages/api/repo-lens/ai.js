/**
 * Server-side AI proxy for the Repo Lens page. Keeps HuggingFace tokens server-side
 * and reaches a local Ollama daemon. Accepts { provider, model, system, user } and
 * returns { text }.
 *
 * Env:
 *   HUGGINGFACE_API_KEY   required for provider "huggingface"
 *   OLLAMA_URL            optional, defaults to http://localhost:11434
 */

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "method not allowed" });
  }

  const { provider, model, system, user } = req.body || {};
  if (!provider || !model || !user) {
    return res.status(400).json({ error: "provider, model, and user are required" });
  }

  try {
    if (provider === "ollama") {
      const text = await callOllama(model, system, user);
      return res.status(200).json({ text });
    }
    if (provider === "huggingface") {
      const text = await callHuggingFace(model, system, user);
      return res.status(200).json({ text });
    }
    return res.status(400).json({ error: `unknown provider ${provider}` });
  } catch (e) {
    return res.status(502).json({ error: e.message || "AI provider call failed" });
  }
}

async function callOllama(model, system, user) {
  const base = process.env.OLLAMA_URL || "http://localhost:11434";
  const r = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      messages: [
        { role: "system", content: system || "" },
        { role: "user", content: user },
      ],
    }),
  });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`Ollama ${r.status}: ${body.slice(0, 200)} (is \`ollama serve\` running with model ${model}?)`);
  }
  const data = await r.json();
  return data.message?.content ?? "";
}

async function callHuggingFace(model, system, user) {
  const key = process.env.HUGGINGFACE_API_KEY;
  if (!key) throw new Error("HUGGINGFACE_API_KEY is not set on the server");
  // Chat-completions style endpoint (router). Falls back cleanly on error.
  const r = await fetch(`https://api-inference.huggingface.co/models/${model}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: system || "" },
        { role: "user", content: user },
      ],
      max_tokens: 512,
      temperature: 0.1,
    }),
  });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`HuggingFace ${r.status}: ${body.slice(0, 200)}`);
  }
  const data = await r.json();
  return data.choices?.[0]?.message?.content ?? "";
}
