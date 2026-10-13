# Third-party notices

AnyPS5 is licensed under the GNU General Public License version 2 only. The macOS arm64 runner (`aps5-fex`) also contains the
following third-party code, which is distributed under the licences reproduced below. The sources are in `3rdparty/FEX`.

## FEX (FEXCore, FEXHeaderUtils, CodeEmitter)

Licence: MIT. Source: https://github.com/FEX-Emu/FEX, tag FEX-2610 with AnyPS5's macOS changes

```
MIT License

Copyright (c) 2019 Ryan Houdek <Sonicadvance1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VIXL (code in FEX's CodeEmitter)

Licence: BSD-3-Clause. Source: https://github.com/Linaro/vixl

```
Copyright 2015, VIXL authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the name of ARM Limited nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## {fmt}

Licence: MIT. Source: https://github.com/fmtlib/fmt

```
Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## xxHash library

Licence: BSD-2-Clause. Source: https://github.com/Cyan4973/xxHash

```
xxHash Library
Copyright (c) 2012-2021 Yann Collet
All rights reserved.

BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Cephes Mathematical Library (128-bit routines in FEX)

Licence: BSD. Source: https://www.netlib.org/cephes/

```
The cephes math library is BSD licensed.
The source can be accessed from https://www.netlib.org/cephes/

Original license from https://www.netlib.org/cephes/readme :
>    Some software in this archive may be from the book _Methods and
> Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
> International, 1989) or from the Cephes Mathematical Library, a
> commercial product. In either event, it is copyrighted by the author.
> What you see here may be used freely but it comes with no support or
> guarantee.
>
>    The two known misprints in the book are repaired here in the
> source listings for the gamma function and the incomplete beta
> integral.
>
>
>    Stephen L. Moshier
>    moshier@na-net.ornl.gov

The author was e-mailed and they allowed it to be relicensed under BSD.
Resources:
https://bugs.gentoo.org/687276
https://lists.debian.org/debian-legal/2004/12/msg00295.html
https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt
https://github.com/nearform/node-cephes/blob/master/LICENSE

E-mail snippit from torch-cephes source:

Return-Path: <steve@moshier.net>
X-Original-To: julien@cornebise.com
Delivered-To: julien@cornebise.com
Received: from atl4mhob11.myregisteredsite.com (atl4mhob11.myregisteredsite.com [209.17.115.49])
    by cornebise.com (Postfix) with ESMTP id D47B139FC0
    for <julien@cornebise.com>; Fri, 25 Oct 2013 16:32:40 +0200 (CEST)
Received: from mailpod1.hostingplatform.com ([10.30.71.116])
    by atl4mhob11.myregisteredsite.com (8.14.4/8.14.4) with ESMTP id r9PEWcwQ003543
    for <julien@cornebise.com>; Fri, 25 Oct 2013 10:32:38 -0400
Received: (qmail 11948 invoked by uid 0); 25 Oct 2013 12:36:20 -0000
X-TCPREMOTEIP: 76.24.25.74
X-Authenticated-UID: steve@moshier.net
Received: from unknown (HELO d510.local) (steve@moshier.net@76.24.25.74)
  by 0 with ESMTPA; 25 Oct 2013 12:36:20 -0000
Date: Fri, 25 Oct 2013 08:36:19 -0400 (EDT)
From: Stephen Moshier <steve@moshier.net>
X-X-Sender: steve@d510
To: Julien Cornebise <julien@cornebise.com>
Subject: Re: Cephes: permission to wrap+distribute for Lua
In-Reply-To: <52653AD3.1010004@cornebise.com>
Message-ID: <alpine.DEB.2.02.1310250827040.17646@d510>
References: <52653AD3.1010004@cornebise.com>
User-Agent: Alpine 2.02 (DEB 1266 2009-07-14)
MIME-Version: 1.0
Content-Type: TEXT/PLAIN; charset=US-ASCII; format=flowed


Julien, thank you for writing.
BSD license is fine, modification is OK.
There are more build scripts available in the web site distributions than
there are on the Netlib.  I think there is an update to Planck's radiation
function that I haven't sent to Netlib yet.  But Netlib is a more stable
site, so it is better to cite that as a reference.


On Mon, 21 Oct 2013, Julien Cornebise wrote:

> -----BEGIN PGP SIGNED MESSAGE-----
> Hash: SHA1
>
> Dear Mr Moshier
>
> I am a researcher in mathematics and machine learning in London, and
> am writing about your awesome Cephes library, whom I found at the
> heart of Scipy.
>
> It is so useful that, with your permission, I would like to wrap it
> for Lua and Torch (a machine learning overlay to Lua, specialized in
> neural nets, see http://www.torch.ch). I would like to distribute it
> as a package for Torch, including your source code along the wrapping
> code.
> This wouldbe a public package, distributed under BSD License. I have
> put a first draft on github:
> https://github.com/jucor/torch-cephes
>
> Hence my three questions, please:
>
> 1/ How would you like to be acknowledged, beyond the comments that are
> already in your code? Do you have any standard header/disclaimer that
> I could add to the documentation?
>
> 2/ At the moment, your code is left untouched. However, if I ever need
> to modify bits of the code, what are the conditions/restrictions?
> Nothing huge -- I definitely do not want to mess with it: I was
> planning to use the natural completion of some functions on the
> completed real line (e.g. CDF returing 1 when called with "infinity",
> or quantiles returning -Infinity when called with 0), either natively
> if supported, or by setting a specific flag  via mtherr().
>
> 3/ I am currently using the source from Netlib. Do you recommend using
> the source from your website instead ?
>
> Thank you very much for your attention,
> and, more importantly, for the time and effort your poured into Cephes.
>
> Best regards,
>
> Julien Cornebise, Ph.D.
> London, UK
> http://www.cornebise.com/julien
> -----BEGIN PGP SIGNATURE-----
> Version: GnuPG v1.4.14 (Darwin)
> Comment: GPGTools - http://gpgtools.org
> Comment: Using GnuPG with Thunderbird - http://www.enigmail.net/
>
> iEYEARECAAYFAlJlOtEACgkQKYR3gC0rw/gIpQCfZKu6+iDh9ghhm6QfsLXnldKN
> BuIAn2zZHu1c/IrRAevhjM7N7xGg0LHO
> =WeP5
> -----END PGP SIGNATURE-----
```

## Berkeley SoftFloat Release 3e

Licence: BSD-3-Clause. Source: http://www.jhauser.us/arithmetic/SoftFloat.html

```
Copyright 2011, 2012, 2013, 2014, 2015, 2017 The Regents of the University of
California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## unordered_dense

Licence: MIT. Source: https://github.com/martinus/unordered_dense

```
MIT License

Copyright (c) 2022 Martin Leitner-Ankerl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## range-v3 (used when configuring FEX)

Licence: BSL-1.0. Source: https://github.com/ericniebler/range-v3

```
========================================================
Boost Software License - Version 1.0 - August 17th, 2003
========================================================

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

==============================================================================
libc++ License
==============================================================================

The libc++ library is dual licensed under both the University of Illinois
"BSD-Like" license and the MIT license.  As a user of this code you may choose
to use it under either license.  As a contributor, you agree to allow your code
to be used under both.

Full text of the relevant licenses is included below.

==============================================================================

University of Illinois/NCSA
Open Source License

Copyright (c) 2009-2014 by the contributors listed in CREDITS.TXT
http://llvm.org/svn/llvm-project/libcxx/trunk/CREDITS.TXT

All rights reserved.

Developed by:

    LLVM Team

    University of Illinois at Urbana-Champaign

    http://llvm.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal with
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimers in the
      documentation and/or other materials provided with the distribution.

    * Neither the names of the LLVM Team, University of Illinois at
      Urbana-Champaign, nor the names of its contributors may be used to
      endorse or promote products derived from this Software without specific
      prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.

==============================================================================

Copyright (c) 2009-2014 by the contributors listed in CREDITS.TXT
  http://llvm.org/svn/llvm-project/libcxx/trunk/CREDITS.TXT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

==============================================================================
Stepanov and McJones, "Elements of Programming" license
==============================================================================

// Copyright (c) 2009 Alexander Stepanov and Paul McJones
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without
// fee, provided that the above copyright notice appear in all copies
// and that both that copyright notice and this permission notice
// appear in supporting documentation. The authors make no
// representations about the suitability of this software for any
// purpose. It is provided "as is" without express or implied
// warranty.
//
// Algorithms from
// Elements of Programming
// by Alexander Stepanov and Paul McJones
// Addison-Wesley Professional, 2009

==============================================================================
SGI C++ Standard Template Library license
==============================================================================

// Copyright (c) 1994
// Hewlett-Packard Company
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  Hewlett-Packard Company makes no
// representations about the suitability of this software for any
// purpose.  It is provided "as is" without express or implied warranty.
//
// Copyright (c) 1996
// Silicon Graphics Computer Systems, Inc.
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  Silicon Graphics makes no
// representations about the suitability of this software for any
// purpose.  It is provided "as is" without express or implied warranty.
//
```

## libusb (libSceUsbd, through libusb-cmake)

Licence: GNU Lesser General Public License 2.1 or later (compatible with GPL-2.0-only). Source: https://github.com/libusb/libusb-cmake (pinned at `bea6567b6`). The full licence text is `3rdparty/libusb-cmake/LICENSE` and `3rdparty/libusb-cmake/libusb/COPYING`.

## Tracy Profiler (only in builds with APS5_ENABLE_TRACY)

Licence: BSD 3-Clause. Source: https://github.com/wolfpld/tracy (pinned at v0.14.1, `30997d5ca`).

```
Tracy Profiler (https://github.com/wolfpld/tracy) is licensed under the
3-clause BSD license.

Copyright (c) 2017-2026, Bartosz Taudul <wolf@nereid.pl>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
