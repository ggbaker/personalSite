---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Remote teaching and seminars with OBS"
subtitle: ""
summary: "Like most grad students entering the job market this year, I've had to
wrestle with the transition to giving talks remotely. Personally, I have
some trouble focusing on virtual seminars since the format tends to be
fairly static: mostly still slides with perhaps a camera window to the
side. To avoid this, I've spent a fair amount of time (*read*:
procrastinating) trying to put together a setup that allows me to move
around and interact with my materials a bit more. Here, I outline an
approach that's worked reasonably well for me."
authors: [admin]
tags: []
categories: []
date: 2020-10-20
lastmod: 2020-10-21
featured: true
draft: false
reading_time: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "A scene from my department job talk"
  focal_point: "Smart"
  preview_only: True

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

{{< figure src="featured.png" title="A scene from my department job talk">}}

# Background

Like most grad students entering the job market this year, I've had to
wrestle with the transition to giving talks remotely. Personally, I have
some trouble focusing on virtual seminars since the format tends to be
fairly static: mostly still slides with perhaps a camera window to the
side. To avoid this, I've spent a fair amount of time (*read*:
procrastinating) trying to put together a setup that allows me to move
around and interact with my materials a bit more. Below, I outline an
approach that's worked reasonably well for me.

Since I'm not teaching this semester, I've mostly focused on how to
adapt an academic seminar to a remote setting, but I would use a
variation of this for remote teaching given the opportunity.

I set out to replicate three particular aspects of in-person
seminars/teaching:
1. The ability to easily handwrite equations/graphs on a blackboard,
2. The ability to draw attention to certain parts of a slide or
   blackboard, whether by laser pointer or physically pointing, and
3. The ability to move and gesture while speaking.

For all of this, I'm indebted to the work of others who took time
to experiment with different methods for remote teaching/seminars and
create guides. In particular,

- Luke Stein's YouTube series: [OBS for teaching tutorial](https://www.youtube.com/watch?v=upTyHsxdlYs&list=PLEPYFCNANvR_ZFZVPp-y_-eo5-AlOK_0Z)
- David J. Malan's post: [Teaching from Home via
Zoom](https://medium.com/@cs50/teaching-from-home-via-zoom-c3b336446fbc)
- A variety of twitter posts by Emily Nix, for example,
  [here](https://twitter.com/EmilyNix100/status/1297261177060302849?s=20)
  and [here](https://twitter.com/EmilyNix100/status/1294068665655025664?s=20)


# Hardware

## Needed harware:

- A computer (obviously): a relatively recent Mac would work best, but
  isn't required (My main desktop runs Linux) 
- A good camera
- A tablet with stylus (iPad with Apple Pencil)
- A green screen and some way to hold it up
- A second monitor is a big plus

I can't overstate the importance of a good camera. If you
have a reasonably modern digital camera, there's a good chance it can
send a video feed over usb to a computer. I use a Nikon D5500 that I
normally use for bird photography, but you almost certainly have a
pretty good webcam already available to you: a smart phone. Prior to
rigging up my Nikon, I simply used my smartphone on a cheap gooseneck
clamp using a free app called
[Droidcam](https://www.dev47apps.com/). A wider angle lens is a big plus
here to give yourself more room to move around without having to stand
too far back. 

Even with a good camera, you need to be extremely well lit to get a good
image (and to get a good green screen effect). I use three lamps each
with some cheap 1500 lumen LED bulbs. 

For the microphone, so long as you don't use your laptop's built in-mic,
I don't think the microphone is that important. The inline mic in most
earbuds works reasonably well, but it might be worth spending a little
bit on the mic. I use a Blue Yeti which works well enough (as a
condenser mic, it works reasonably well even sitting a bit back from
it), but if I had to do it again, I probably would have gotten a cheap
lav mic.

Of course, to put yourself in the same scene with slides, there's not
much substitute for a proper green screen. I got a cheap one on Amazon
for under $20, but be aware that the cheap screens come folded and need
steam-ironing to remove the creases. Ideally, you'll have a wall behind
you to mount it, but otherwise you'll need some kind of stand (which
will add some expense).

Lastly, you'll need some way to write easily. For me, that means an iPad
with the Apple Pencil. You will also need some way to get the video feed
from the tablet into your computer. 

Finally, it's extremely useful (though not necessary) to have at least
one extra monitor so you can have the chat window and participants'
videos visible separately.

## Room setup

{{< figure src="messydesk.jpg" title="A very messy desk set up for presenting" >}}

I sit on a bar stool about 1.5 meters back from my camera in the left
third of the camera's field of view. This leaves most of the camera's
field of view as empty green screen that I can fill with the slides and
gesture over. In the image below showing the OBS setup, the red box is
the extent of the area I can gesture in. My camera lens is fairly wide
(28mm full-frame equivalent, roughly an 80 degree field of view). If
you're using a narrower camera angle---most smart phones are probably
closer to 60 degrees---you'd have to sit farther back from the camera to
get a similar effect. If you're considering buying a webcam, look for
one with a wider field of view.

Since I'm sitting so far back from my desk, I also use a small table for
my iPad, and I adjust my microphone to be as close as possible.

{{< figure src="roomsetup.jpg" title="Where I sit" >}}

# Software

## Needed software

- [OBS](https://obsproject.com/)
- Virtual Cam Extension for OBS
  ([Windows](https://obsproject.com/forum/resources/obs-virtualcam.949/),
  [Mac](https://github.com/johnboiles/obs-mac-virtualcam),
  [Linux](https://github.com/CatxFish/obs-v4l2sink))
- PDF software of choice on for the tablet (I use [Readdle
  Documents](https://readdle.com/documents) on the iPad)
- Some way to get the video feed from the tablet to the computer (see below)

## OBS setup

{{< figure src="obs.png" title="OBS Scene arrangement">}}

Since many others have given excellent tutorials on how to use OBS (see
Luke Stein's tutorial linked above), I'll skip the details of the setup
and jump straight to the scene arrangement.

My primary scene in OBS has three sources:

1. Webcam input, with Chroma Key (green screen) filter applied, and with
   the image reflected (so I can more easily see where I'm "pointing")
2. Flat color as a background (black worked best for me)
3. Video input from my iPad (see below)

In order to make better use of the space in the OBS canvas, I also
adjusted the aspect ratio of my beamer slides to be a bit more square
(12x10 worked will for me). This can be accomplished using the
`beamerposter` package:

```
\usepackage[orientation=landscape,size=custom,width=12,height=10,scale=0.5,debug]{beamerposter}
```

## Video input from tablet

The biggest challenge of all this was finding a way to mirror my iPad's
display to my desktop so OBS could use it. If you have a relatively
recent Mac, this is easy since recent Macs and iPads by default support
a system called [Sidecar](https://support.apple.com/en-us/HT210380) that
allows you to use your iPad as a second screen (with touch support) for
your Mac.

Since my main desktop isn't a Mac, I settled on using Apple's
[Airplay](https://www.apple.com/airplay/) protocol to wirelessly mirror
my iPad display to my desktop; however, this isn't supported by default
(Airplay is meant to mirror to a device like an AppleTV). Fortunately,
with appropriate software, a regular computer can become an Airplay
receiver. On Linux, this can be accomplished with an open source tool
called [UxPlay](https://github.com/antimof/UxPlay). I know similar
software exists for Windows, but all options I saw were paid software,
so I'm hesitant to recommend one in particular without trying them
first.


# Zoom setup

Finally, we just need to get everything into Zoom. There's two ways to
do this, each with it's own advantages.

## Webcam

The most straightforward method is to simply use the OBS virtual cam
extension (link above). Within Zoom, I'd recommend specifically
"spotlighting" your camera so the view doesn't switch to another
person's camera when they speak.

Besides simplicity, this method is pretty much sure to work even if
you're not using Zoom (say, Skype or Blackboard Collaborate)---so long
as whatever system you're using supports a webcam.

The main issue I had with this method is that Zoom's compression can
make the slide text difficult to read at times unless the internet
connection for everyone is absolutely perfect. 

This method also has the advantage that it's likely to work with other
software such as Skype or Blackboard Collaborate.

## Screen sharing

I eventually settled on using the Screen Share feature with the
"optimize for video" setting enabled.

First, I create a window with the OBS canvas output
(right click on the OBS canvas -> Fullscreen projector). I then share
this window with Zoom. Once screen sharing, I then disable my webcam in
zoom to avoid duplicate images.

This method *mostly* fixes the blurry text issue. Unfortunately, the
video feed can lag the audio a bit which can be distracting for
the viewers. This lag is somewhat unpredictable; in one of my practice
talks it was apparently up to a second, and in my final department talk it
wasn't much of an issue at all. Your mileage may vary. 

I haven't tested this method with non-Zoom conference software, but I
wouldn't expect it to work well there. It seems like most software uses
a compression optimized for relatively static scenes for screen sharing,
so movement would likely appear very choppy to viewers. It only works in
Zoom because of Zoom's "optimize for video option" for screen sharing.

# The final result

{{< figure src="examplescene.gif" title="The final result" >}}

Overall, the whole setup is a bit of a house of cards, but it mostly
replicated what I felt were the most important aspects of in-person
talks.

Although I haven't had a chance to use this for teaching yet, I expect
things would work similarly well, either using a note-taking app on the
iPad for a pure "chalk-and-talk" sort of lecture or using a hybrid
approach mixing slides with handwriting.

# Update: Some extra considerations

One thing I forgot to mention: before trying any of this in an actual
talk, it's very important to test it out with other people. You can't
see exactly how Zoom or whatever software you're using compresses the
video, so it's important to check with other people whether the material
you're showing is legible to others. 

Also, as I mentioned a bit before, the exact way Zoom handles video can
be a bit unpredictable, so it's good to be prepared with a backup option
in case things unexpectedly go wrong. In my case, I have a backup "Scene
Collection" in OBS that is a more traditional arrangement of the slides
with a smaller camera overlay in the corner, and in an emergency I can
switch to that in about a minute. 

