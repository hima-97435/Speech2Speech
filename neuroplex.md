This diagram is the Neuroplex speech-to-speech architecture. The important idea is that although it contains ASR → LLM → speech-generation components, the main path does not actually convert the input speech into text and then feed that text to the LLM. Instead, it passes continuous hidden representations between modules. The text tokens shown on the top/right are mainly for debugging.

1. First, understand the whole flow

Think of it as:

Your voice
   ↓
Audio features
   ↓
ASR
   ↓
ASR hidden representation
   ↓
ASR2LLM Adapter
   ↓
LLM
   ↓
LLM hidden representation
   ↓
LLM2T2C Adapter
   ↓
Text2Codes
   ↓
Speech codes
   ↓
Codes2Audio
   ↓
Your response audio

There is also:

Conditioning Audio
       ↓
Conditioning Encoder
       ↓
     T2C

The conditioning audio is used to provide information to the speech-generation side, such as characteristics of the desired output voice/style.

The architecture explicitly separates understanding, reasoning, and speech synthesis, while adapters translate representations between these specialized modules.

2. Let's do an actual dry run

Suppose I speak:

"What is the capital of India?"

We start with:

audio = microphone.read()

Conceptually:

audio
 ↓
[waveform samples]

For example:

[0.02, 0.04, 0.01, -0.03, ...]

This is just raw audio.

The model cannot directly give this giant waveform to the ASR transformer.

3. Feature Extractor

The first component converts the waveform into useful acoustic features.

features = feature_extractor(audio)

Conceptually:

Raw audio
    ↓
Feature Extractor
    ↓
audio features

Instead of:

0.02, 0.04, 0.01, -0.03, ...

we now have something more like:

[
  [0.12, -0.31, 0.72, ...],
  [0.15, -0.29, 0.68, ...],
  [0.18, -0.25, 0.70, ...],
  ...
]

Each row represents a portion of the audio.

The paper describes these features as a tensor:

a ∈ R[B × Tin × Din]

where roughly:

B = batch size
Tin = number of input time steps
Din = feature dimension

4. Feature Extractor → ASR

Now:

asr_hidden = ASR(features)

The ASR consists of an encoder/decoder transformer.

Important point:

The ASR doesn't simply produce text and stop.

It internally creates a rich representation:

ASR hidden states
        ↓
 ┌────────────────────┐
 │ semantic information│
 │ phonetic information│
 │ acoustic information│
 │ temporal information│
 └────────────────────┘

So instead of immediately doing:

audio → "What is the capital of India?"

the main path keeps:

audio → ASR hidden representation

The ASR can still produce:

"What is the capital of India?"

through its ASR head.

But that's the debug path.

5. What are the "ASR Debug Tokens"?

Look at the green boxes:

ASR Head
   ↓
ASR Debug Tokens

These allow us to inspect what the ASR representation means.

For example:

ASR Debug Tokens:

["What", "is", "the", "capital", "of", "India", "?"]

This is extremely useful for debugging.

You can ask:

Did the model understand what the user said?

If the user said:

"What is the capital of India?"

and ASR debug tokens say:

"What is the capital of India?"

then the ASR side is working.

But these tokens don't have to be passed to the LLM.

That's one of the most important ideas in this architecture.

6. ASR2LLM Adapter — the most important bridge

Now we have:

ASR hidden representation
          ↓
      ASR2LLM
          ↓
LLM-compatible representation

In pseudo-code:

llm_input = asr2llm(asr_hidden)

Why do we need this?

Because:

ASR representation ≠ LLM representation

Suppose:

ASR:

[B, 120, 512]

while the LLM expects something like:

[B, 80, 4096]

You can't simply do:

llm(asr_hidden)

because the representation spaces are different.

The adapter learns:

ASR space
   ↓
   mapping
   ↓
LLM space

The paper formally describes this as:

xa2l = ga2l(xasr)

where the adapter converts ASR hidden states into LLM-compatible embeddings.

7. Now the LLM enters

We now execute:

llm_hidden = LLM(llm_input)

This is where actual language understanding/reasoning happens.

Input:

"What is the capital of India?"

Conceptually the LLM understands:

Question type:
    factual question

Entity:
    India

Requested information:
    capital

Answer:
    New Delhi

But again, internally we aren't necessarily passing around the literal string:

"What is the capital of India?"

We are passing embeddings/hidden states.

The LLM produces something like:

LLM hidden representation
        ↓
[semantic representation of the response]

The paper calls this:

xllm = fLLM(xa2l)

8. LLM Debug Tokens

The LLM also has a debug head:

LLM hidden state
      ↓
   LLM Head
      ↓
LLM Debug Tokens

For example:

["The", "capital", "of", "India", "is", "New", "Delhi", "."]

This lets researchers inspect:

"What answer did the LLM internally produce?"

Again, this is primarily an inspection/debugging point, not necessarily the representation passed to T2C.

The paper reports that these debug tokens aligned with the content of the generated response in its experiments.

9. LLM2T2C Adapter

Now we need to move from:

LLM representation

to:

speech-generation representation

So:

t2c_input = llm2t2c(llm_hidden)

Conceptually:

LLM latent space
       ↓
   LLM2T2C
       ↓
T2C latent space

Again, this is a translation between representation spaces.

The paper describes:

xl2t = gl2t(xllm)

10. Now comes the interesting part: Text2Codes

This module is slightly confusing because of its name.

You might think:

Text2Codes

means:

text → speech

But that's not exactly what happens here.

Its job is:

representation
      ↓
acoustic codes

The T2C module is an autoregressive sequence model that generates discrete speech codes.

For example, conceptually:

LLM/T2C representation
        ↓
T2C
        ↓
[182, 91, 204, 72, 81, ...]

These aren't words.

They are compressed representations of speech.

Think of them as:

speech_code_1
speech_code_2
speech_code_3
speech_code_4
...

The codes contain information needed to reconstruct speech.

11. Why don't we directly generate waveform from the LLM?

Because waveform is extremely high-dimensional.

Imagine generating:

0.012
0.017
0.021
0.015
...

for thousands of audio samples.

That's inefficient for an autoregressive model.

Instead:

LLM
 ↓
compact speech representation
 ↓
speech codes
 ↓
decoder
 ↓
waveform

This makes the generative problem much smaller.

12. Conditioning Audio

Now look at the bottom-left:

Conditioning Audio
       ↓
Conditioning Encoder
       ↓
      T2C

This is separate from the user's input speech.

Think of conditioning audio as:

"Here is what the generated speech should sound like."

For example, suppose we provide a reference voice:

Conditioning Audio
=
5-second recording of desired speaker

The conditioning encoder extracts information about that audio.

Conceptually:

reference voice
      ↓
conditioning encoder
      ↓
voice/style representation

Then T2C can use:

                ┌──────────────┐
LLM2T2C ───────→│              │
                │     T2C      │
Conditioning ──→│              │
                └──────┬───────┘
                       ↓
                  speech codes

So T2C is deciding something like:

"What should be said?" + "How should it sound?"

and produces acoustic codes.

13. Finally: Codes2Audio

This is the component you specifically asked about.

We now have:

Speech Codes
    ↓
Codes2Audio
    ↓
Waveform

The paper describes C2A as a streaming convolutional decoder that reconstructs audio waveforms from the discrete speech codes.

This is basically a decoder.

14. What does Codes2Audio actually do?

Suppose T2C generates:

C = [c1, c2, c3, c4, c5, ...]

where each ci is a compressed representation of some small section of speech.

Codes2Audio learns approximately:

f(C) → waveform

So:

[182, 91, 204, 72, 81, ...]
             ↓
        Codes2Audio
             ↓
[0.01, 0.03, 0.05, 0.02, -0.01, ...]

That final sequence is the actual audio waveform.

Then your speaker/audio device plays it.

15. Think of Codes2Audio like an image decoder

This analogy is useful.

Imagine:

Image
 ↓
Image Encoder
 ↓
compressed latent
 ↓
Image Decoder
 ↓
Image

Similarly:

Speech
 ↓
Speech representation / codes
 ↓
compressed acoustic codes
 ↓
Codes2Audio
 ↓
Speech waveform

So Codes2Audio isn't doing the reasoning.

It isn't deciding:

"What should I say?"

That's the LLM/T2C side.

Codes2Audio is basically saying:

"Given these acoustic codes, reconstruct the actual sound."

16. Why is it convolutional?

Because reconstructing audio from compressed representations is very similar to what audio codecs/decoders do.

A convolutional decoder can progressively increase temporal resolution.

Conceptually:

Low-resolution representation

[c1][c2][c3][c4]
       ↓
   Conv/upsample
       ↓
[c1][ ][c2][ ][c3][ ][c4][ ]
       ↓
   Conv/upsample
       ↓
[c1][ ][ ][ ][c2][ ][ ][ ]...
       ↓
   waveform

The actual architecture is more sophisticated, but this is the intuition.

The C2A module is specifically designed as a streaming decoder, allowing audio to be reconstructed incrementally rather than waiting for the entire response.