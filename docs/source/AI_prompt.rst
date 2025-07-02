
Script with AI
==============

How to Use This Guide
---------------------

1. Copy/download the entire content of :download:`AI_prompt.md </AI_prompt.md>` below
2. Paste it into your AI chatbot or let your agent (`Aider <https://aider.chat/>`_ / `Cline <https://cline.bot/>`_ / ...) read it
3. Example prompts:

   .. code-block:: markdown

      Create a λpic simulation with:
      - 2D laser-plasma interaction
      - 1024x1024 grid
      - Gaussian laser (a0=5, w0=2μm)
      - Plasma density 10nc
      - Field diagnostics every 100 steps

      Convert this WarpX/EPOCH input to λpic:
      [paste input file]

      Fix this error:
      [paste error message]

Best Practices
--------------

- Include physical units in your prompts
- Specify dimensionality (2D/3D) 
- Multi-turn conversation, gradually refine your simulation


Full AI Prompt Content
----------------------

.. literalinclude:: AI_prompt.md
   :language: markdown
