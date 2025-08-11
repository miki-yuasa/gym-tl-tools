Installation
============

Requirements
------------

- Python >=3.10, <3.14
- NumPy >= 1.21
- Gymnasium >= 0.26
- Pydantic ^2.0
- spottl ^2.13

Installing from PyPI
--------------------

You can install ``gym-tl-tools`` using pip:

.. code-block:: bash

   pip install gym-tl-tools

Installing from Source
----------------------

If you are developing locally, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/miki-yuasa/gym-tl-tools.git
   cd gym-tl-tools
   pip install -e .

Using UV
--------

If you prefer using UV for dependency management:

.. code-block:: bash

   git clone https://github.com/miki-yuasa/gym-tl-tools.git
   cd gym-tl-tools
   uv sync

For development with additional dependencies (including documentation tools):

.. code-block:: bash

   uv sync --group dev
