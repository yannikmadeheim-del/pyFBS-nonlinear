pyFBS is an ongoing open source project and everybody is welcome to contribute to:
 * asking questions (https://gitlab.com/pyFBS/pyFBS_support/-/issues)
 * reporting bugs (https://gitlab.com/pyFBS/pyFBS/-/issues)
 * feature requests (https://gitlab.com/pyFBS/pyFBS/-/issues)
 * adding new code (by creating Merge Request)

Asking questions
----------------
Asking questions and making them public is extremely important for anyone 
who will ever encounter a similar problem/dilemma. 
The community of developers and other users of the package will answer your question 
and thus help both developers and other users.

To ask a question, please create an issue on our support page https://gitlab.com/pyFBS/pyFBS_support/-/issues. 
You can also write us an email at info.pyfbs@gmail.com.

Reporting bugs
--------------
``pyFBS`` is a library that is still evolving, and sometimes we come across unexpected, not intuitive code behaviour. 
If you encounter such a bug, please report it by opening the issue at https://gitlab.com/pyFBS/pyFBS/-/issues and mark them with the Bug label.  
To make it easier to reproduce the bug and consequently solve it, please submit a minimal working example, 
support the problem with a screenshot or sample files.

Feature requests
----------------
We will be pleased with suggestions for improving the current and introducing new functionalities in ``pyFBS``. 
Please, post them by opening the issue at https://gitlab.com/pyFBS/pyFBS/-/issues and mark them with the New feature label. 
Add a brief description of the proposed feature and what are its benefits. 
In addition, you can support your suggestions with pictures or links to relevant references.

Adding new code
---------------
Contribution to ``pyFBS`` can also be made by adding new code or documentation.
Before adding a new code, please open the issue with the appropriate label (New feature, Documentation). 
This allows us to determine in the discussion whether the proposed subject fits pyFBS, 
whether the proposed topic may already be under development 
and assign the users who will work on the proposed topic.

Cloning
^^^^^^^
Before starting writing your own code you have to download the latest version of the pyFBS library by running:

.. code-block:: 

    git clone https://gitlab.com/pyFBS/pyFBS.git
    cd pyFBS
    python -m pip install -e .

or you can fork the repository and clone it from your account.

Creating new branch
^^^^^^^^^^^^^^^^^^^
New code is always added via a new branch. 
Please, use an informative and descriptive branch name. 
You can also use the number of the issue resolved by this branch (like iss5).

.. code-block:: 

    git branch iss5

Coding
^^^^^^
Now that you are in the new branch, you can start making changes to the repository. 
When adding new functionalities, try to follow the current code structure. 
Add new code to the most content-related file. 
If the new functionality does not belong to any of the existing files, during the discussion in the open issue, 
agree on how it would make sense to implement this functionality.

Code style
**********
Code should follow the philosophy of `Python programin language <https://en.wikipedia.org/wiki/Python_(programming_language)#Design_philosophy_and_features>`_:
 - Beautiful is better than ugly.
 - Explicit is better than implicit.
 - Simple is better than complex.
 - Complex is better than complicated.
 - Readability counts.

The naming and code layout convention should follow the PEP 8. 
The exception is line widths are permitted to go to 100 characters, or even more.

Adding additional comments between lines of code is very welcome, as it greatly increases the intelligibility of the code.

Documentation
^^^^^^^^^^^^^

Good documentation is a crucial feature of any good library. 
With a brief description of the function's content, input and output parameters and an example of its use, we enable users to use this function correctly.

Documentation style
*******************

Every function must have a docstring as in the following example. Docstrings are defined inside ``""" """``. 
First, provide a brief introduction of the function. 
Then the input parameters are presented using ``:parm parameter_name:`` command, followed by a short description of the parameters.
We can define also the type of the input parameter using ``:parm parameter_name:`` command, followed by the variable type.
At the end of the list of all input parameters, also the output of the function is defined using the ``:return:`` command.
Type of the output is defined by ``:rtype:``
At the end of the docstring, we can also provide an example of usage of the function and expected result. 

.. code-block:: python

    def my_function(my_param_1, my_param_2):
        """
        Returns sum of my_param_1 and my_param_2.
        
        :parm my_param_1: the first parameter of summation
        :type my_param_1: array
        :parm my_param_2: the second parameter of summation
        :type my_param_2: array
        
        :return: sum of ``my_param_1`` + ``my_param_2``
        :rtype: array
        
        Example: 
        >>>a = my_function_1(1, 1)
        >>>print(a)
        2
        """
        result = my_param_1 + my_param_2
        return result

Notebook examples
*****************
``pyFBS`` library has a collection of notebooks, where most of the functionalities are used on simple examples. 
Please find the appropriate existing notebook or create a new one inside the ``.\examples\`` folder to add the example of added functionlities. 
These notebooks are also part of the testing procedure, so make sure that all notebooks run without errors.

Online documentation
********************
Documentation, displayed at https://pyfbs.readthedocs.io/en/latest/, is located at ``.\docs\``.
These pages present the theoretical background of methods included in ``pyFBS``. 
All files are in from of `Restructured Text (reST) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

At the beginning of every topic is an introduction followed by the most important equations and relevant references. 
In the end, we provide an extensive description of how to use the introduced topic using the ``pyFBS`` functionalities. 

Testing
^^^^^^^
After making changes, please test changes locally before creating a merge request.
The code testing is fully automated. To test the code, you have to install the ``tox`` library:

.. code-block:: 

    pip install tox

Testing code
************
Once the ``tox`` is installed, you just have to runt the ``tox.ini`` script using the command:

.. code-block:: 

    cd pyFBS
    tox

The ``tox`` script will create a virtual environment and test all notebook examples and also all tests defined inside folder ``.\test\``.

Testing documentation
*********************
Documentation is tested sepparatly, by running commands:

.. code-block:: 

    cd doc
    make clean
    make html

Generated documentation is saved to folder ``.\_build\html``. 
Here you can open HTML pages in the browser and see the effects of your changes. 

Creating Merge Request
^^^^^^^^^^^^^^^^^^^^^^
When the changes pass all local tests, it is time to create a merge request.
When creating a merge request, add a short description and assign code reviewers, who will check the changes and accept the merge.
Creating a merge request will automatically run continuous integration (CI) testing. 
If a merge request is solving one or more of the issues, mention this in the description of the merge request using ``Closes #4, #6``.
This command will automatically close listed issues, once the branch will be merged. 
More useful commands are listed here: https://docs.gitlab.com/ee/user/project/issues/managing_issues.html
