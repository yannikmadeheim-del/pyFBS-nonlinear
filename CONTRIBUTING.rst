pyFBS is an ongoing open-source project and everybody is welcome to contribute to:
 * asking questions (https://gitlab.com/pyFBS/pyFBS_support/-/issues),
 * reporting bugs (https://gitlab.com/pyFBS/pyFBS/-/issues),
 * feature requests (https://gitlab.com/pyFBS/pyFBS/-/issues),
 * adding new code (by creating Merge Request).

Asking questions
----------------
Asking questions and making them public is immensely helpful for anyone 
who will ever encounter a similar problem or dilemma. 
The community of developers and other ``pyFBS`` users will try to answer your question 
which will benefit both developers and users.

To ask a question, please create an issue on our support page https://gitlab.com/pyFBS/pyFBS_support/-/issues. 
You can also write to us via email at info.pyfbs@gmail.com.

Reporting bugs
--------------
``pyFBS`` is a still-evolving library and that's why you might encounter some unexpected and unintuitive code behavior. 
If you encounter such a bug, please report it by opening an issue at https://gitlab.com/pyFBS/pyFBS/-/issues and mark it with the Bug label.  
To make it easier for the developers to reproduce the bug in order to fix it, please submit a minimal working example
(e.g. support the problem with a screenshot or sample files).

Feature requests
----------------
We will welcome suggestions for improving existing and introducing new functionalities to the ``pyFBS``. 
Please post them by opening the issue at https://gitlab.com/pyFBS/pyFBS/-/issues and mark it with the New feature label. 
Add a brief description of the proposed feature and outline its benefits. 
In addition, you can equip your suggestions with pictures or links to relevant references.

Adding new code
---------------
Contribution to ``pyFBS`` can also be made by adding new code or writing documentation on the existing one.
Before adding a new code, please open the issue with the appropriate label (New feature, Documentation). 
This allows us to determine through the discussion whether the proposed subject fits ``pyFBS``, 
if the proposed topic is not already under development, 
and also to assign the users who will work on the topic.

Cloning
^^^^^^^
Before starting writing your own code you have to download the latest version of the ``pyFBS`` library by running:

.. code-block:: 

    git clone https://gitlab.com/pyFBS/pyFBS.git
    cd pyFBS
    python -m pip install -e .

You can also fork the repository and clone it from your account.

Creating new branch
^^^^^^^^^^^^^^^^^^^
New code is always added via a new branch. 
Please use an informative and descriptive branch name. 
As a branch name, you can also use the number of the issue you are trying to resolve (like iss5).

.. code-block:: 

    git branch iss5

Coding
^^^^^^
Once you create your own branch, you can start making changes to the repository. 
When adding new functionalities, try to follow the current code structure. 
You should always add new code to the most content-related file. 
If the new functionality does not relate to any of the existing files, start a discussion in the open issue 
on how it would make sense to implement this functionality.

Code style
**********
Code should follow the philosophy of the `Python programming language <https://en.wikipedia.org/wiki/Python_(programming_language)#Design_philosophy_and_features>`_:
 - Beautiful is better than ugly.
 - Explicit is better than implicit.
 - Simple is better than complex.
 - Complex is better than complicated.
 - Readability counts.

The naming and code layout convention should follow the PEP 8. 
The exception is line widths which are permitted to exceed 79 characters.

Adding additional comments between lines of code is most welcome, as it greatly increases the intelligibility of the code.

Documentation
^^^^^^^^^^^^^

Consistent documentation is a crucial feature of any great library. 
A brief description of the function's content, input and output parameters, and an example of its use 
allows other users to implement this function correctly.

Documentation style
*******************

Every function must have a docstring as demonstarted in the following example.  

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

Docstrings are defined inside ``""" """``. 
First, provide a brief introduction of the function. 
Then, the input parameters are listed using ``:parm parameter_name:`` command, followed by a short description of the parameters.
It is advisable to define the type of the input parameter using ``:parm parameter_name:`` command, followed by the variable type.
At the end of the list of all the input parameters, the output of the function is defined using the ``:return:`` command.
Type of the output is defined by ``:rtype:``
The docstring can end with an example demonstrating the implementation of the function and expected result.

Notebook examples
*****************
``pyFBS`` library has a collection of notebooks in which most of the functionalities are depicted using simple examples. 
If you want to add the example of new functionalities, please find the appropriate existing notebook or create a new 
one inside the ``.\examples\`` folder. 
These notebooks are also a part of the testing procedure, so make sure that all notebooks run without errors.

Online documentation
********************
Documentation, displayed at https://pyfbs.readthedocs.io/en/latest/, is located at ``.\docs\``.
These pages include the theoretical background of methods included in ``pyFBS``. 
All files are in form of `Restructured Text (reST) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

At the beginning of each topic there is an introduction of the method followed by the fundamental equations and relevant references. 
In the end, an extensive description of how to use the introduced topic using the ``pyFBS`` functionalities is provided. 

Testing
^^^^^^^
After making the changes, please test changes locally first before creating a merge request.
The code testing is fully automated. To test the code, you have to install the ``tox`` library:

.. code-block:: 

    pip install tox

In addition, it is necessary to install requirements for developers, listed in ``requirements_dev.txt``, using the command:

.. code-block:: 

    pip install -r requirements_dev.txt 

Testing code
************
Once the ``tox`` is installed, you just have to run the ``tox.ini`` script using the command:

.. code-block:: 

    cd pyFBS
    tox

The ``tox`` script will create a virtual environment and will test all notebook examples and all tests that are defined inside ``.\test\`` folder.

Testing documentation
*********************

Before building the documentation, execute the following command:

.. code-block:: 

    cd doc
    pip install -r requirements_dev.txt 

Documentation is tested separately by running commands:

.. code-block:: 

    make clean
    make html

Generated documentation can be found in ``.\_build\html``. 
Here you can open HTML pages in your browser and see your changes. 

Creating Merge Request
^^^^^^^^^^^^^^^^^^^^^^
When the changes pass all local tests, it's' time to create a merge request.
When creating a merge request, add a short description and assign code reviewers that will check the changes and accept the merge.
Creating a merge request will automatically run the continuous integration (CI) testing. 
If a merge request resolves one or more issues, mention this in the description of the merge request using ``Closes #4, #6``.
This will automatically close mentioned issues once the branch will be merged. 
More useful commands can be found here: https://docs.gitlab.com/ee/user/project/issues/managing_issues.html
