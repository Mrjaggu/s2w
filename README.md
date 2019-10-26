# s2w
A reusable library that can convert a paragraph of spoken english to written english.



## 1 Always make sure you can acquire the data once again — and your team know how to do it

## Requirement

Now, let’s take a look at requirements.txt
numpy
pandas
scikit-learn
List of required packages is very simple but fine for now. It is very important to track all project’s dependencies. Later we will see that simply listing required packages is not enough, but if you are not using even this simple list, please start.

## 2 List all your dependencies and create separate environment
For every project create separate conda/virtual environment and list all dependencies. If your project requires some additional non-python dependencies (like database drivers, system-wide packages) list them all explicitly, or if possible create installation scripts for them. Remember, the number of manual steps must be as limited as possible. This also applies to “one-time activities”. I’m sometimes surprised how often those one-timers are executed.

## Creating Python package
First step is to create a package. There are many befits with basically no cost. Some of them:<br>
Clear project structure<br>
Ability to create source/binary distribution for deployment<br>
Versioning<br>
Automatic dependency checking<br>
Easy creation of extensions<br>
Separation (e.g. you might not want include training code for prediction environment)<br>
And more :)<br>

## 3 Package your code
<br>
The last thing is to create setup.py in the project root directory. This file contains all meta-information about the package. It can be very large, but for our purposes (at this moment) a few lines are sufficient:
<br>
Very last thing to do is to install our package. To do this type following (in project root):
pip install -e .
There is no magic: instead of package name we provide a dot which means “install current directory”. The -e switch makes it installed in development mode. This basically means that every change in package source will be immediately reflected in installed version — there would be no need to reinstall it after making changes. You can see your package after executing pip list:
