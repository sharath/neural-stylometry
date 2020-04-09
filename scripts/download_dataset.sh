#!/bin/sh

wget -O maildir.tar.gz  https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz
tar -zxvf maildir.tar.gz
rm maildir.tar.gz 
mv maildir dataset