#!/bin/sh

cd /tmp \
&& rm -fr hotpants \
&& git clone https://github.com/acbecker/hotpants.git \
&& cd hotpants \
&& make \
&& cp hotpants /usr/local/bin/ \
&& cd .. \
&& rm -fr hotpants \
&& echo "HOTPANTS successfully installed"
