# Commercial licensing

Oído has two parts, under two kinds of license:

- **The code** (engine, firmware, tools) is free software under the **GNU GPL v3**. You may use, modify
  and ship it under those terms. For consumer devices this includes providing the corresponding source code and the
  information needed to install modified firmware.
- **The model files** (`models/`) are licensed file by file ([`models/LICENSE-WEIGHTS`](models/LICENSE-WEIGHTS)). The GPL
  on the code does not extend to them.
  - `nemo8.tnm`, `nemo4.tnm`, `oido_stream.tnm`, `oido_v2a.tnm`, `oido_v2b.tnm`, `oido_es.tnm`, `nemo_lm.tlm`,
    `oido_v2.tlm`, `oido_es.tlm` and `oido_es/tokenizer.model` are **non-commercial**: CC BY-NC-SA 4.0 plus
    [our terms of use](models/TERMS.md). Using them in a product, a paid service or internal business use needs a
    written license from Lokutor. So do files derived from them.
  - `nemo_small/tokenizer.model` is NVIDIA's file, unmodified, under CC-BY-4.0.
  - From the commit that adds this text, these files are released under CC BY-NC-SA 4.0 with Lokutor's terms. Copies
    released before it keep the license they were released under (CC-BY-SA-4.0 for `nemo4.tnm`, `oido_stream.tnm`,
    `oido_v2a.tnm`, `oido_v2b.tnm`; CC-BY-4.0 for the others).

If those terms don't fit your product, [Lokutor](https://lokutor.com) (Hashing Works, S.L.L., Madrid) offers:

- **Commercial licenses** for the engine and firmware, without GPL obligations, priced per device or as an annual license.
- **Commercial licenses for the non-commercial model files** above, per device or as an annual license.
- **Additional models:** Catalan, Basque, Galician and more languages (Spanish is released in this repository);
  an int4 profile for other languages; domain and vocabulary adaptation.
- **A full on-device voice stack** for the same chip: speech-to-text, text-to-speech ([Ito](https://github.com/lokutor-ai/ito))
  and turn-taking.
- **Integration and support** on your hardware.

## Let's collaborate

We would rather see Oído used than sit on a shelf. If you are a small company, a startup, a maker selling small
batches, a school or a research group, tell us what you are building. We are happy to work something out, including
no-cost licenses.

Contact: contact@lokutor.com
