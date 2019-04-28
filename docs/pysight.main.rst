---------------------------------
Running PySight (``main`` module)
---------------------------------

Generally you'll want to use `pysight.main.run` as the
standard\\default way to run this application. It can be
run without any arguments, which will open the :ref:`GUI <gui>`,
or run with a pre-existing configuration file name.

The other two functions allow you to run PySight on several
list files one after the other, either in a sequential manner
(:func:`pysight.main.run_batch_lst`) or in a parallel manner
(:func:`pysight.main.mp_batch`).

.. autofunction:: pysight.main.run

.. autofunction:: pysight.main.run_batch_lst

.. autofunction:: pysight.main.mp_batch
