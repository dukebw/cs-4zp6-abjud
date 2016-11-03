Steps to run the website with Python: (most likely will use a different framework later that is similar to this, but this one is really easy to use)
These steps are meant for Linux systems

Basic Requirements:
1)Python 2.7 (tensorflow might be risky with python 3+)
2) pip => python package manager
    -->for windows, it comes with python 2.7 or 3+, link: http://stackoverflow.com/questions/4750806/how-do-i-install-pip-on-windows



(optional, worst case) virtualenv => for setting up a python version if your locally installed python does not work with tornado
    -> this is very easy to install and use with pip. will put up the info later

3)Tornado => the actual web framework


here is a link detailing the steps for fedora:
http://tutorialforlinux.com/2015/03/08/how-to-getting-started-with-python-tornado-on-fedora-linuxgnu-easy-guide/


For windows (can't gaurantee because i havent tried it):
https://pypi.python.org/pypi/tornado
--> you can download the .whl file that best fits your computer

http://stackoverflow.com/questions/27885397/how-do-i-install-a-python-package-with-a-whl-file
 --> how to install the .whl file



then, all you'd need to do is navigate to the directory with the <app>.py file and run "python <app>.py"
this is because the tornado framework gets installed in python's "lib" folder so it will be available for import (the <app>.py file imports the framework)

if this web app does not work if you set up tornado this way, using a 'virtualenv' works with no problems
