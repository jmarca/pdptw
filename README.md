# exploring pdptw with Google OR Tools

In python

So Google has an excellent OR tools suite, available on github
at <http://github.com/google/or-tools>.  I downloaded and compiled the
code, but wasn't sure how to install it such that the python libraries
would work (documentation of that is poor in the repo).  So while I
also have a working copy of the main google OR tools codebase, the
python in this directory is using the release version of the python
tools.

To install, follow the instructions given on the OR tools
documentation website, <https://developers.google.com/optimization/>.
Specifically, I followed the instructions
at
<https://developers.google.com/optimization/introduction/installing.html#unix_binary_python>  Because
I run slackware and nothing is ever as easy as `sudo apt get out of
jail`, I just kept trying and then installing the missing
dependencies.  Not really difficult.

In addition to the base python Google OR tools library, for the code
in this directory to run you will need to also install NumPy (numpy
1.12.1) and MatPlotLib (matplotlib 2.0.0), both of which I installed
with pip.
