When you map:

`placeholder.bohacek.wang` → `placeholder.github.io`

**it is NOT “just a redirect.”**

It becomes a **full custom domain** for your GitHub Pages site.

Here is the exact difference:

---

# 1. Redirect vs. Full Domain Mapping

### A. Redirect (NOT what GitHub Pages uses)

A redirect is when the browser changes the URL:

`placeholder.bohacek.wang`
→ you get sent to
`placeholder.github.io`
and the address bar shows the GitHub URL.

This is *not* what you are setting up.

---

### B. Full Domain Mapping (what you are doing)

With a **CNAME DNS record**, your subdomain *becomes* the official domain of the site.

The browser URL **stays**:

```
https://placeholder.bohacek.wang
```

GitHub serves your website under this hostname as if it were a standalone domain.

■ You get HTTPS via GitHub (Let’s Encrypt).
■ SEO treats it as your domain.
■ GitHub Pages uses your `CNAME` file to bind your site to it.

This is why you must create the `CNAME` file in the GitHub repo.

---

# 2. Summary

| Action                                | What Happens                                                 |
| ------------------------------------- | ------------------------------------------------------------ |
| **Redirect**                          | User ends up on `placeholder.github.io`                      |
| **CNAME Domain Mapping** (your setup) | Your custom domain is the real site address; GitHub hosts it |

Your setup results in **a real custom subdomain**, not a redirect.

---

# 3. Want both root and subdomains?

You can also connect:

* `bohacek.wang`
* `www.bohacek.wang`
* `placeholder.bohacek.wang`
* etc.

Just tell me your plan and I can give you exact DNS entries for Dynadot.

Would you like your **main site** on your root domain as well?
