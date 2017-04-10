using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using TextToMotionWeb.Data;
using Microsoft.AspNetCore.Authorization;
using TextToMotionWeb.Models;
using Microsoft.AspNetCore.Identity;
using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace TextToMotionWeb.Controllers
{
    public class DashboardController:Controller
    {
        [Authorize]
        public IActionResult Index()
        {
            return View();
        }

        [Authorize]
        public IActionResult ProcessImage()
        {
            return View();
        }

        [Authorize]
        public IActionResult ProcessVideo()
        {
            return View();
        }

        [Authorize]
        public IActionResult ProcessStream()
        {
            return View();
        }
    }
}
