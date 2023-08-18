#!/bin/sh

cd /tmp \
&& echo "** Getting HOTPANTS sources from GitHub into /tmp/hotpants **\n" \
&& rm -fr hotpants \
&& git clone https://github.com/acbecker/hotpants.git \
&& cd hotpants \
&& echo "\n** Patching the sources **\n" \
&& { patch -p1 <<'EOF'
diff --git a/Makefile b/Makefile
index 7e86f33..2ab7fa3 100644
--- a/Makefile
+++ b/Makefile
@@ -27,7 +27,7 @@ LIBDIR        =  ../../cfitsio/lib

 # standard usage
 # recently added -std=c99 after a bug report
-COPTS = -funroll-loops -O3 -ansi -std=c99 -pedantic-errors -Wall -I$(CFITSIOINCDIR) -D_GNU_SOURCE
+COPTS = -funroll-loops -fcommon -O3 -ansi -std=c99 -pedantic-errors -Wall -I$(CFITSIOINCDIR) -D_GNU_SOURCE
 LIBS  = -L$(LIBDIR) -lm -lcfitsio

 # compiler
diff --git a/alard.c b/alard.c
index 2467b9f..9d2f3b2 100644
--- a/alard.c
+++ b/alard.c
@@ -1,7 +1,7 @@
 #include<stdio.h>
 #include<string.h>
 #include<math.h>
-#include<malloc.h>
+/*#include<malloc.h>*/
 #include<stdlib.h>
 #include<fitsio.h>

diff --git a/extractkern.c b/extractkern.c
index 9857930..edcf3a5 100644
--- a/extractkern.c
+++ b/extractkern.c
@@ -2,7 +2,7 @@
 #include<string.h>
 #include<strings.h>
 #include<math.h>
-#include<malloc.h>
+/*#include<malloc.h>*/
 #include<stdlib.h>
 #include<fitsio.h>

diff --git a/functions.c b/functions.c
index 1dab79c..f9d29b0 100644
--- a/functions.c
+++ b/functions.c
@@ -1,7 +1,7 @@
 #include<stdio.h>
 #include<string.h>
 #include<math.h>
-#include<malloc.h>
+/*#include<malloc.h>*/
 #include<stdlib.h>
 #include<fitsio.h>
 #include<ctype.h>
diff --git a/main.c b/main.c
index 2d8d44c..0564b2b 100644
--- a/main.c
+++ b/main.c
@@ -1,7 +1,7 @@
 #include<stdio.h>
 #include<string.h>
 #include<math.h>
-#include<malloc.h>
+/*#include<malloc.h>*/
 #include<stdlib.h>
 #include<fitsio.h>
 #include<ctype.h>
diff --git a/maskim.c b/maskim.c
index e5c25a2..a9d1b46 100644
--- a/maskim.c
+++ b/maskim.c
@@ -3,7 +3,7 @@
 #include<string.h>
 #include<strings.h>
 #include<math.h>
-#include<malloc.h>
+/*#include<malloc.h>*/
 #include<stdlib.h>
 #include<fitsio.h>
EOF
} \
&& echo "\n** Compiling the sources **\n" \
&& make \
&& echo "\n** Compilation was successful **" \
&& echo "** Installing it to /usr/local/bin **\n" \
&& cp hotpants /usr/local/bin/ \
&& cd .. \
&& echo "\n** Cleaning up **\n" \
&& rm -fr hotpants \
&& echo "HOTPANTS successfully installed"
